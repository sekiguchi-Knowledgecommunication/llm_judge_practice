from dotenv import load_dotenv
import os
from openevals.llm import create_llm_as_judge
from openevals.prompts import CORRECTNESS_PROMPT

load_dotenv()
os.environ["OPENAI_API_KEY"] = os.getenv("OPENAI_API_KEY")

correctness_evaluator = create_llm_as_judge(
    prompt=CORRECTNESS_PROMPT,
    model="openai:gpt-5-mini",
)

inputs = """
    静的な展示意味
    """
# These are fake outputs, in reality you would run your LLM-based system to get real outputs
outputs = """
    「静的な展示」とは、動きや変化が少なく、一つの場所に固定されている展示物のことを指します。
    これには、絵画、彫刻、説明パネル、写真、資料などが含まれます。
    対照的に、動的な展示は、インタラクティブな要素や動きのある展示物を含みます。
    """
reference_outputs ="""
    「ポイントは「観客の参加や操作を前提とする」展示に対し、観客が受動的に鑑賞する展示を指す言葉になります。\n
    候補となる表現\n\n
    非インタラクティブ展示 (Non-interactive exhibition)\n
    最も直接的な対義語で、観客が操作や参加をせず、展示物を一方的に見たり聞いたりするだけの形式。\n
    例：絵画や写真の展覧会、ガラスケース内の博物館展示。\n\n
    静的展示 (Static exhibition)\n
    時間や観客の行動によって変化しない展示を強調する場合に用いる。\n
    例：模型や標本の陳列。\n\n
    受動的展示 (Passive exhibition)\n
    観客の関与が「能動」ではなく「受動」であることを前面に出す表現。\n
    例：映像をただ鑑賞するシアター型展示。
    """
# When calling an LLM-as-judge evaluator, parameters are formatted directly into the prompt
eval_result = correctness_evaluator(
  inputs=inputs,
  outputs=outputs,
  reference_outputs=reference_outputs
)

print(eval_result)