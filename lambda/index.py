import json
import os
import boto3
import re
import requests  # requestsライブラリを追加
from botocore.exceptions import ClientError

# Lambda コンテキストからリージョンを抽出する関数
def extract_region_from_arn(arn):
    match = re.search('arn:aws:lambda:([^:]+):', arn)
    if match:
        return match.group(1)
    return "us-east-1"  # デフォルト値

# グローバル変数としてクライアントを初期化
bedrock_client = None

# モデルID
MODEL_ID = os.environ.get("MODEL_ID", "us.amazon.nova-lite-v1:0")
# FastAPI サーバー
FASTAPI_URL = os.environ.get("FASTAPI_URL", "http://localhost:8501")

def lambda_handler(event, context):
    try:
        # 従来のBedrock初期化コード
        global bedrock_client
        if bedrock_client is None:
            region = extract_region_from_arn(context.invoked_function_arn)
            bedrock_client = boto3.client('bedrock-runtime', region_name=region)
            print(f"Initialized Bedrock client in region: {region}")
        
        print("Received event:", json.dumps(event))
        
        # Cognitoで認証されたユーザー情報を取得
        user_info = None
        if 'requestContext' in event and 'authorizer' in event['requestContext']:
            user_info = event['requestContext']['authorizer']['claims']
            print(f"Authenticated user: {user_info.get('email') or user_info.get('cognito:username')}")
        
        # リクエストボディの解析
        body = json.loads(event['body'])
        message = body['message']
        conversation_history = body.get('conversationHistory', [])
        
        print("Processing message:", message)
        
        # 会話履歴を使用
        messages = conversation_history.copy()
        
        # ユーザーメッセージを追加
        messages.append({
            "role": "user",
            "content": message
        })
        
        # 会話履歴から完全なプロンプトを構築
        # FastAPIサーバーはシンプルなプロンプト文字列を期待するため、会話履歴を適切なフォーマットに変換
        full_prompt = ""
        for msg in messages:
            role_prefix = "ユーザー: " if msg["role"] == "user" else "アシスタント: "
            full_prompt += f"{role_prefix}{msg['content']}\n"
        
        # 最後にシステムからの指示を追加
        full_prompt += "アシスタント: "
        
        # FastAPI用のリクエストペイロード
        request_payload = {
            "prompt": full_prompt,
            "max_new_tokens": 512,
            "temperature": 0.7,
            "top_p": 0.9,
            "do_sample": True
        }
        
        print(f"Calling FastAPI server at {FASTAPI_URL}/generate with payload:", json.dumps(request_payload))
        
        # FastAPI サーバーに接続
        response = requests.post(
            f"{FASTAPI_URL}/generate",
            json=request_payload
        )
        
        # HTTPエラーをチェック
        response.raise_for_status()
        
        # レスポンスを解析
        response_body = response.json()
        print("FastAPI response:", json.dumps(response_body, default=str))
        
        # 応答の検証
        if not response_body.get('generated_text'):
            raise Exception("No response content from the model")
        
        # アシスタントの応答を取得
        assistant_response = response_body['generated_text']
        
        # アシスタントの応答を会話履歴に追加
        messages.append({
            "role": "assistant",
            "content": assistant_response
        })
        
        # 成功レスポンスの返却
        return {
            "statusCode": 200,
            "headers": {
                "Content-Type": "application/json",
                "Access-Control-Allow-Origin": "*",
                "Access-Control-Allow-Headers": "Content-Type,X-Amz-Date,Authorization,X-Api-Key,X-Amz-Security-Token",
                "Access-Control-Allow-Methods": "OPTIONS,POST"
            },
            "body": json.dumps({
                "success": True,
                "response": assistant_response,
                "conversationHistory": messages
            })
        }
        
    except requests.exceptions.RequestException as e:
        print(f"FastAPI request error: {str(e)}")
        return {
            "statusCode": 500,
            "headers": {
                "Content-Type": "application/json",
                "Access-Control-Allow-Origin": "*",
                "Access-Control-Allow-Headers": "Content-Type,X-Amz-Date,Authorization,X-Api-Key,X-Amz-Security-Token",
                "Access-Control-Allow-Methods": "OPTIONS,POST"
            },
            "body": json.dumps({
                "success": False,
                "error": f"FastAPI request failed: {str(e)}"
            })
        }
    except Exception as error:
        print("Error:", str(error))
        
        return {
            "statusCode": 500,
            "headers": {
                "Content-Type": "application/json",
                "Access-Control-Allow-Origin": "*",
                "Access-Control-Allow-Headers": "Content-Type,X-Amz-Date,Authorization,X-Api-Key,X-Amz-Security-Token",
                "Access-Control-Allow-Methods": "OPTIONS,POST"
            },
            "body": json.dumps({
                "success": False,
                "error": str(error)
            })
        }