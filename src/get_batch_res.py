from openai import OpenAI

import os
import pandas as pd
import numpy as np 
import json
from dotenv import load_dotenv



if __name__ == "__main__":
    load_dotenv()

    OPENAI_KEY = os.getenv("OPENAI_KEY")
    client = OpenAI(
        api_key=OPENAI_KEY
    )

    file_response = client.files.content("file-sFhfTtVQ2bLEEoTCeJasyjMI")
    print(file_response.text)