# Intent Server
🚀 Intent Server is an advanced intent classification engine that categorizes user queries into predefined intents. It can also prompt follow-up questions for ambiguous queries, enhancing user interaction and understanding.

## Features
* 🎯 Intent Classification: Accurately classifies user queries into specific intents.
* 🤖 Follow-up Questions: Asks clarifying questions when queries are ambiguous.
* 🔧 Customizable: Easily configurable settings for different use-cases.

## Setup Guide
Follow these steps to set up the Intent Server on your local system.

### 1. Clone the Repository
First, clone this repository to your local system:
```commandline
git clone https://github.com/yourusername/intent-server.git
cd intent-server
```

### 2. Create a Conda Environment (Highly Recommended)
Create a new Conda environment for the project:
```commandline
conda create --name intent-server-env python=3.11
conda activate intent-server-env
```

### 3. Install Dependencies
Install all necessary dependencies using the `requirements.txt` file:
```commandline
pip install -r requirements.txt
```

### 4. Configure Settings
Configure the settings in the `setup.yaml` file. All parameters are optimized for general use-case, but you can tweak them as needed.

### 5. API Tokens
If you are using OpenAI or another external LLM provider, provide your API token in the `.env` file. The `.env` file is already created for you; just add your token.

### 6. Activate the Server
Run the following command to activate the API endpoint:
```commandline
python activate_server.py
```

The API endpoint will be available at `/intent-server/chatgpt/p5`. To configure hosting settings, refer to the `setup.yaml` file.

### Configuration Files
`setup.yaml`: Contains all the configurable parameters for the server.
`.env`: Stores your API tokens for external LLM providers.


---

📌 For additional information or bug reports, please open an issue. Feedback is welcome!