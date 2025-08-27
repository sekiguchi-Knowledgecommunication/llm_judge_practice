import os
import traceback
import sys
from dotenv import load_dotenv
import pandas as pd
from langchain_openai import ChatOpenAI
from langchain_anthropic import ChatAnthropic
from langchain_google_genai import ChatGoogleGenerativeAI
from openevals.llm import create_llm_as_judge
from openevals.prompts import CORRECTNESS_PROMPT
from datasets import ethics_questions, all_questions, get_dataset_info

load_dotenv()
os.environ["OPENAI_API_KEY"] = os.getenv("OPENAI_API_KEY")

EVAL_MODEL = "openai:o3-mini"

load_dotenv()

os.environ["OPENAI_API_KEY"]

def init_models():
    print("モデルの初期化を開始...")
    models = {}
    
    # 比較対象のモデル
    # 新規にモデルを定義する場合はこのdictに追記する
    model_configs = [
        {"name": "gpt-4o", "class": ChatOpenAI, "params": {"model": "gpt-4o", "temperature": 1.0}},
        {"name": "gpt-5-mini", "class": ChatOpenAI, "params": {"model": "gpt-5-mini", "temperature": 1.0}},
        {"name": "gpt-5", "class": ChatOpenAI, "params": {"model": "gpt-5", "temperature": 1.0}},
    ]
    
    for config in model_configs:
        model_name = config["name"]
        try:
            models[model_name] = config["class"](**config["params"])
            print(f"  {model_name}の初期化成功")
        except Exception as e:
            print(f"  {model_name}の初期化エラー: {e}")
            print(f"  エラーの詳細:\n{traceback.format_exc()}")
    
    if not models:
        print("警告: すべてのモデルの初期化に失敗しました。")
        sys.exit(1)
    
    print(f"初期化完了: {', '.join(models.keys())}")
    return models

# 評価用データセットの情報を取得
dataset_info = get_dataset_info()


print("評価者モデルを初期化中...")
try:
    correctness_evaluator = create_llm_as_judge(
        prompt=CORRECTNESS_PROMPT,
        model=EVAL_MODEL,
    )
    print(f"評価者モデルの初期化完了: {EVAL_MODEL}")
except Exception as e:
    print(f"評価者モデルの初期化エラー: {e}")
    print(f"エラーの詳細:\n{traceback.format_exc()}")
    print("評価者モデルの初期化に失敗したため、処理を中止します。")
    sys.exit(1)


def evaluate_models(models, questions):
    print("\n===== モデル評価プロセス開始 =====")
    results = []
    total_questions = len(questions)
    
    for i, question_data in enumerate(questions, 1):
        question = question_data["question"]
        reference = question_data["reference"]
        
        for model_name, model in models.items():
            print(f"  モデル '{model_name}' で回答を生成中...")
            try:
                # モデルからの回答を取得
                response = model.invoke(question)
                model_answer = response.content
                print(f"  回答生成完了 (文字数: {len(model_answer)})")
                
                # 回答を評価
                print(f"  評価者モデルで回答を評価中...")
                try:
                    eval_result = correctness_evaluator(
                        inputs=question,
                        outputs=model_answer,
                        reference_outputs=reference
                    )
                    
                    print(f"  評価結果: {eval_result}")
                    
                    results.append({
                        "model": model_name,
                        "question": question,
                        "answer": model_answer,
                        "evaluation": eval_result
                    })
                    
                except Exception as eval_error:
                    print(f"  評価処理中にエラーが発生しました: {eval_error}")
                    print(f"  評価エラーの詳細:\n{traceback.format_exc()}")
            
            except Exception as e:
                error_type = type(e).__name__
                print(f"  回答生成中にエラーが発生しました ({model_name}): {error_type}: {e}")
                print(f"  エラーの詳細:\n{traceback.format_exc()}")
    
    print("\n全ての質問の評価が完了しました。結果を集計中...")
    df = pd.DataFrame(results)
    return df


def main():
    print("\n===== メイン処理開始 =====")
    
    print("モデルを初期化中...")
    models = init_models()
    
    print("モデル評価を開始...")
    results_df = evaluate_models(models, all_questions)
    
    print("\n===== 評価結果の分析 =====")
    
    print(f"\n評価結果の概要:")
    print(f"総評価数: {len(results_df)}")
    print(f"評価対象モデル: {', '.join(results_df['model'].unique())}")
    
    print("\nCSVファイルに詳細結果を保存中...")
    export_df = results_df.copy()
    export_df['evaluation'] = export_df['evaluation'].apply(lambda x: str(x))
    export_df.to_csv("llm_evaluation_results.csv", index=False)
    print("詳細な結果は 'llm_evaluation_results.csv' に保存されました。")
    
    print("\n===== 評価プロセス完了 =====")
    return results_df


if __name__ == "__main__":
    print("\n===== スクリプト実行開始 =====")
    results = main()
