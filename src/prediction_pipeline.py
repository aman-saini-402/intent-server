import json
import os
from typing import Any

import pandas as pd
import torch
import yaml
from dotenv import load_dotenv
from langchain.callbacks.manager import CallbackManagerForLLMRun
from langchain.chains import SequentialChain, TransformChain, LLMChain
from langchain.chains.base import Chain
from langchain.chat_models import AzureChatOpenAI
from langchain.llms.base import LLM
from langchain.prompts import PromptTemplate
from torch.nn import functional as F
from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    logging
)

from src.utils import json_parser

logging.set_verbosity_error()

# Load project settings
with open("setup.yaml", "r") as file:
    config = yaml.load(file, yaml.Loader)

# Load secret keys
load_dotenv()


def load_prompt(prompt_file):
    with open(f"prompts/{prompt_file}", 'r') as f:
        return f.read()


class Pipeline:
    def __init__(self):
        self.local_prediction_chain = self._get_local_prediction_chain()
        self.evaluation_chain = self._get_gpt_eval_chain()
        self.detector_chain = self._get_gpt_detector_chain()
        self.condense_chain = self._get_gpt_condense_question_chain()
        self.oos_followup_question_chain = self._get_gpt_oos_followup_question_chain()
        self.amb_followup_question_chain = self._get_gpt_amb_followup_question_chain()

    def _get_local_prediction_chain(self) -> Chain:
        """
        Function to initialize a local llm (bert like) prediction chain.
        This will load the model, wrap a langchain compatible API and use this model to create prediction chain
        """
        # Specify compute specs
        run_on = "cuda" if torch.cuda.is_available() else "cpu"

        # Load data
        df = pd.read_csv("data/preprocessed_data.csv")

        # Define label 2 id mappings
        labels = df["category"].unique().tolist()
        label2id = {row["category"]: row["category_id"] for idx, row in
                    df.drop_duplicates(["category", "category_id"]).iterrows()}
        id2label = {row["category_id"]: row["category"] for idx, row in
                    df.drop_duplicates(["category", "category_id"]).iterrows()}

        # Create tokenizer
        tokenizer = AutoTokenizer.from_pretrained(
            f"data/{config['FINETUNE']['MODEL_CHECKPOINT']}" + "_ft",
            add_prefix_space=True
        )

        # Load model
        model = AutoModelForSequenceClassification.from_pretrained(
            f"data/{config['FINETUNE']['MODEL_CHECKPOINT']}" + "_ft",
            num_labels=len(labels),
            id2label=id2label,
            label2id=label2id
        ).to(run_on)

        # Create langchain wrapper around local llm
        class LLMWrap(LLM):
            """
            A wrapper around local_llm to be used in langchain
            """

            @property
            def _llm_type(self) -> str:
                return "localLLM"

            def _call(
                    self,
                    prompt: str,
                    stop: list[str] | None = None,
                    run_manager: CallbackManagerForLLMRun | None = None,
            ) -> str:
                if stop is not None:
                    raise ValueError("stop kwargs are not permitted.")

                # Inference
                inputs = tokenizer.encode(prompt, return_tensors="pt").to(
                    run_on)  # moving to mps for Mac (can alternatively do 'cpu')
                logits = model(inputs).logits
                sorted_predictions = torch.sort(logits, 1, descending=True).indices.tolist()[0]
                predicted_intent = id2label[sorted_predictions[0]]
                top_k_predictions = [id2label[i] for i in
                                     sorted_predictions[:config["PREDICTION_PIPELINE"]["TOP_K_PREDICTIONS"]]]
                probability_scores = F.softmax(logits, dim=-1).cpu().numpy()[0]
                confidence = float(max(probability_scores))
                to_return = {
                    "intent": predicted_intent,
                    "confidence": confidence,
                    "top_k_intents": top_k_predictions
                }

                # Return result
                return json.dumps(to_return)

        # JSON parser for adding extra parameter to the chain output
        def json_parser(inputs: dict) -> dict:
            result = json.loads(inputs["output"])
            inputs["intent"] = result["intent"]
            inputs["confidence"] = result["confidence"]
            inputs["top_k_intents"] = result["top_k_intents"]
            return inputs

        transform_chain = TransformChain(
            input_variables=["query", "output"], output_variables=["intent", "confidence", "top_k_intents"],
            transform=json_parser
        )

        # Prepare prediction chain
        local_prediction_chain_inter = LLMChain(
            llm=LLMWrap(),
            prompt=PromptTemplate(template="{query}", input_variables=["query"]),
            output_key="output"
        )
        local_prediction_chain = SequentialChain(
            chains=[local_prediction_chain_inter, transform_chain],
            input_variables=["query"],
            output_variables=["intent", "confidence", "top_k_intents"]
        )

        return local_prediction_chain

    def _get_gpt_eval_chain(self) -> Chain:
        """
        Create evaluation chain.
        This chain is usually applied after prediction chain.
        """
        # Load data - get intent descriptions
        df_labels = pd.read_csv("data/intent_desc_3p_pov.csv")
        labels_with_desc = ""
        for _, row in df_labels.iterrows():
            labels_with_desc += f"label- {row['intent']} ; Meaning- {row['description']}\n"

        # Evaluation prompt
        intent_eval_prompt = PromptTemplate.from_template(
            load_prompt("evaluation_prompt.txt"),
            partial_variables={"labels": labels_with_desc}
        )

        # Prepare evaluation chain
        openai_llm = AzureChatOpenAI(
            temperature=config["PREDICTION_PIPELINE"]["EVAL_MODEL_TEMP"],
            max_tokens=400,
            openai_api_key=os.environ["AZURE_OPENAI_API_KEY"],
            deployment_name=config["PREDICTION_PIPELINE"]["EVAL_DEPLOYMENT_NAME"],
            openai_api_version="2023-05-15",
            openai_api_base=config["PREDICTION_PIPELINE"]["EVAL_AZURE_API_BASE"]
        )
        evaluation_chain_openai = LLMChain(
            llm=openai_llm,
            prompt=intent_eval_prompt,
            output_key="evaluation"
        )

        return evaluation_chain_openai

    def _get_gpt_detector_chain(self) -> Chain:
        """
        Create a detection chain.
        This chain helps in detection of ambiguous/out-of-scope user query.
        """
        # Load data - intent description
        df_labels = pd.read_csv("data/intent_desc_3p_pov.csv")
        labels_with_desc = ""
        for _, row in df_labels.iterrows():
            labels_with_desc += f"label- {row['intent']} ; Meaning- {row['description']}\n"

        # Detection prompt
        ambiguity_detection_prompt = PromptTemplate.from_template(
            load_prompt("ambiguity_detection_prompt.txt"),
            partial_variables={"labels": labels_with_desc}
        )

        # Prepare detection chain
        openai_llm = AzureChatOpenAI(
            temperature=config["PREDICTION_PIPELINE"]["DETECT_MODEL_TEMP"],
            max_tokens=100,
            openai_api_key=os.environ["AZURE_OPENAI_API_KEY"],
            deployment_name=config["PREDICTION_PIPELINE"]["DETECT_DEPLOYMENT_NAME"],
            openai_api_version="2023-05-15",
            openai_api_base=config["PREDICTION_PIPELINE"]["DETECT_AZURE_API_BASE"]
        )
        evaluation_chain_openai = LLMChain(
            llm=openai_llm,
            prompt=ambiguity_detection_prompt,
            output_key="detection",
            verbose=False
        )

        return evaluation_chain_openai

    def _get_gpt_condense_question_chain(self) -> Chain:
        """
        Create a question condenser chain.
        This chain consolidate two or more query into a standalone query without losing context.
        """
        # Condenser prompt
        query_condense_prompt = PromptTemplate.from_template(
            load_prompt("question_condenser_prompt.txt")
        )

        # Prepare condenser chain
        openai_llm = AzureChatOpenAI(
            temperature=config["PREDICTION_PIPELINE"]["CONDENSE_MODEL_TEMP"],
            max_tokens=400,
            openai_api_key=os.environ["AZURE_OPENAI_API_KEY"],
            deployment_name=config["PREDICTION_PIPELINE"]["CONDENSE_DEPLOYMENT_NAME"],
            openai_api_version="2023-05-15",
            openai_api_base=config["PREDICTION_PIPELINE"]["CONDENSE_AZURE_API_BASE"]
        )
        condense_chain_openai = LLMChain(
            llm=openai_llm,
            prompt=query_condense_prompt,
            output_key="condensation",
            verbose=False
        )

        return condense_chain_openai

    def _get_gpt_oos_followup_question_chain(self) -> Chain:
        """
        Create a followup question chain for out of scope inputs.
        This chain generate a context aware followup question when user input is out of scope.
        """
        # Load data - get intent descriptions
        df_labels = pd.read_csv("data/intent_desc_3p_pov.csv")
        labels_with_desc = ""
        for _, row in df_labels.iterrows():
            labels_with_desc += f"label- {row['intent']} ; Meaning- {row['description']}\n"

        # followup question generation prompt
        followup_question_prompt = PromptTemplate.from_template(
            load_prompt("oos_followup_question_prompt.txt"),
            partial_variables={"labels": labels_with_desc}
        )

        # Prepare followup question chain
        openai_llm = AzureChatOpenAI(
            temperature=config["PREDICTION_PIPELINE"]["FOLLOWUP_GEN_MODEL_TEMP"],
            max_tokens=400,
            openai_api_key=os.environ["AZURE_OPENAI_API_KEY"],
            deployment_name=config["PREDICTION_PIPELINE"]["FOLLOWUP_GEN_DEPLOYMENT_NAME"],
            openai_api_version="2023-05-15",
            openai_api_base=config["PREDICTION_PIPELINE"]["FOLLOWUP_GEN_AZURE_API_BASE"]
        )
        followup_chain_openai = LLMChain(
            llm=openai_llm,
            prompt=followup_question_prompt,
            output_key="followup_question",
            verbose=False
        )

        return followup_chain_openai

    def _get_gpt_amb_followup_question_chain(self) -> Chain:
        """
        Create a followup question chain for ambiguous inputs.
        This chain generate a context aware followup question when user input is ambiguous.
        """
        # Load data - get intent descriptions
        df_labels = pd.read_csv("data/intent_desc_3p_pov.csv")
        labels_with_desc = ""
        for _, row in df_labels.iterrows():
            labels_with_desc += f"label- {row['intent']} ; Meaning- {row['description']}\n"

        # followup question generation prompt
        followup_question_prompt = PromptTemplate.from_template(
            load_prompt("ambiguous_followup_question_prompt.txt"),
            partial_variables={"labels": labels_with_desc}
        )

        # Prepare followup question chain
        openai_llm = AzureChatOpenAI(
            temperature=config["PREDICTION_PIPELINE"]["FOLLOWUP_GEN_MODEL_TEMP"],
            max_tokens=400,
            openai_api_key=os.environ["AZURE_OPENAI_API_KEY"],
            deployment_name=config["PREDICTION_PIPELINE"]["FOLLOWUP_GEN_DEPLOYMENT_NAME"],
            openai_api_version="2023-05-15",
            openai_api_base=config["PREDICTION_PIPELINE"]["FOLLOWUP_GEN_AZURE_API_BASE"]
        )
        followup_chain_openai = LLMChain(
            llm=openai_llm,
            prompt=followup_question_prompt,
            output_key="followup_question",
            verbose=False
        )

        return followup_chain_openai

    async def _eval_layer(self, query: str, intent: str) -> tuple[dict, str]:
        """
        Create evaluation layer using evaluation chain
        """
        # Call evaluation chain
        eval_result = await self.evaluation_chain.acall({"query": query, "intent": intent})
        eval_result = eval_result["evaluation"]

        # Parse llm output
        eval_result = json_parser(eval_result)

        # Extract suggested intent
        if eval_result["accuracy"] == "YES":
            return eval_result, ""

        new_intent = eval_result["alternate_intent"]
        return eval_result, new_intent

    async def run(self, query: str, query_2: str | None = None, followup_q: str | None = None) -> dict[str: Any]:
        to_return = {
            "final_prediction": "000x999",
            "is_evaluated": False,
            "followup_question": "00xx_no_followup_question_xx00",
            "final_input_to_bert": "",
            "is_oos": False
        }

        while True:
            if not query_2:
                # Bert only prediction
                bert_result = self.local_prediction_chain(query)
                to_return["final_confidence"] = bert_result["confidence"]
                to_return["final_top_k_intents"] = bert_result["top_k_intents"]
                to_return["final_input_to_bert"] = query

                # Check confidence
                if bert_result["confidence"] > config["PREDICTION_PIPELINE"]["CONFIDENCE_THRESHOLD"]:
                    to_return["final_prediction"] = bert_result["intent"]
                    return to_return

                # Call detector
                detector_result = await self.detector_chain.acall({"query": query})
                detector_result = json_parser(detector_result["detection"])

                # Check Ambiguity
                input_category = detector_result["Predicted Category"]
                if input_category == "straightforward":
                    # Run evaluation chain - since it's a low confidence prediction from BERT
                    eval_result, new_intent = await self._eval_layer(query, bert_result["intent"])
                    to_return["is_evaluated"] = True
                    if eval_result["accuracy"] == "YES":
                        to_return["final_prediction"] = bert_result["intent"]
                    else:
                        to_return["final_prediction"] = new_intent
                    return to_return
                elif input_category == "ambiguous":
                    # Generate followup question
                    followup_result = self.amb_followup_question_chain(query)
                    to_return["followup_question"] = followup_result["followup_question"]
                    return to_return
                elif input_category == "out_of_scope":
                    # Generate followup question
                    followup_result = self.oos_followup_question_chain(query)
                    to_return["followup_question"] = followup_result["followup_question"]
                    to_return["is_oos"] = True
                    return to_return
                else:
                    msg = "Predicted category of input is not one of [straightforward, ambiguous, out_of_scope]. This prediction is performed by detector LLM."
                    raise RuntimeError(msg)
            else:
                # Condense Question
                condense_result = await self.condense_chain.acall(
                    {"query": query,
                     "followup_question": followup_q,
                     "query_2": query_2})

                # pass generated standalone question to BERT
                query = condense_result["condensation"]
                query_2 = ""
