from math import inf

from langchain_core.runnables import chain
from langchain_openai import ChatOpenAI

from schemas.mcq.context import ForwardContext
from schemas.mcq.request import McqRequest
from utils.constants import CHOICE_TOKENS


def build_remote_forward():
    import os

    import dotenv

    dotenv.load_dotenv()
    LLAMA_CPP_SERVER_URL = os.getenv("LLAMA_CPP_SERVER_URL")

    model = ChatOpenAI(
        base_url=LLAMA_CPP_SERVER_URL,
        api_key="NOT_NEED",
        name="CHOOSE_NUMBER",
        temperature=0,
        extra_body={
            "max_tokens": 1,
            "grammar": 'root ::= ("1" | "2" | "3" | "4" | "5")',
            "n_probs": 50,
            "min_keep": 5,
        },
    )

    @chain
    def forward(data: McqRequest) -> ForwardContext:
        output = model.invoke(data["messages"])
        len_choices = data["len_choices"]
        top = output.response_metadata["logprobs"]["content"][0]["top_logprobs"]
        top_dict = {tok["token"]: tok["logprob"] for tok in top if tok["token"] in CHOICE_TOKENS}
        target = [top_dict.get(str(i), -inf) for i in range(1, len_choices + 1)]

        return {"data": data, "score": target}

    return forward
