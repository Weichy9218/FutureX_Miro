import os
from openai import OpenAI
from dotenv import load_dotenv

load_dotenv()

client = OpenAI(
    api_key=os.getenv("OPENROUTER_API_KEY"),
    base_url=os.getenv("OPENROUTER_BASE_URL"),
)

def list_models():
    """列出所有可用模型"""
    print("=" * 60)
    print("可用模型列表")
    print("=" * 60)
    models = client.models.list()
    model_ids = [m.id for m in models.data]
    
    def company_key(model_id: str):
        if "/" in model_id:
            company = model_id.split("/", 1)[0]
            return (0, company.lower(), model_id.lower())
        return (1, "", model_id.lower())
    
    for mid in sorted(model_ids, key=company_key):
        print(mid)
    print(f"\n总共 {len(model_ids)} 个模型\n")

def test_model_classification(model_name: str):
    """测试模型是否能正常进行领域分类"""
    print("=" * 60)
    print(f"测试模型: {model_name}")
    print("=" * 60)
    
    test_prompts = [
        "Will Biden win the 2024 election?",
        "Who will win the NBA championship this year?",
        "What is the current Bitcoin price?",
    ]
    
    domains = ["politics", "sports", "crypto", "culture", "finance", 
               "business", "technology", "weather", "health", "space"]
    
    for i, prompt in enumerate(test_prompts, 1):
        print(f"\n测试 {i}/3: {prompt}")
        try:
            completion = client.chat.completions.create(
                model=model_name,
                messages=[
                    {"role": "system", "content": "You are a precise domain classifier. Respond only with the domain name."},
                    {"role": "user", "content": f"""Classify this question into ONE domain from: {', '.join(domains)}

Question: {prompt}

Respond with only the domain word:"""}
                ],
                temperature=0.0,
                max_tokens=50,
            )
            result = completion.choices[0].message.content
            
            # 检查返回内容是否为空
            if result is None or result.strip() == "":
                print(f"⚠️  模型回复为空！")
                return False
            
            result = result.strip()
            print(f"✅ 模型回复: {result}")
        except Exception as e:
            print(f"❌ 错误: {e}")
            return False
    
    print(f"\n✅ 模型 {model_name} 测试通过！")
    return True

def test_simple_chat(model_name: str):
    """简单的对话测试"""
    print("=" * 60)
    print(f"简单对话测试: {model_name}")
    print("=" * 60)
    try:
        # 注意：某些模型（如 gpt-5）在特定代理上可能对 max_tokens 有最小值限制
        # 使用 50 而不是 10 以避免触发这些限制
        completion = client.chat.completions.create(
            model=model_name,
            messages=[
                {"role": "user", "content": "Say 'Hello' in one word."},
            ],
            temperature=0.5,
            max_tokens=50,
        )
        result = completion.choices[0].message.content
        
        # 检查返回内容是否为空
        if result is None or result.strip() == "":
            print(f"⚠️  模型回复为空！")
            return False
        
        print(f"✅ 模型回复: {result}")
        return True
    except Exception as e:
        print(f"❌ 错误: {e}")
        return False

if __name__ == "__main__":
    import sys
    
    # 1. 如果带参数 --list，则列出所有模型
    if len(sys.argv) > 1 and sys.argv[1] == "--list":
        list_models()
        sys.exit(0)
    
    # 2. 推荐的模型（按性价比和速度）
    recommended_models = [
        "openai/gpt-5",                      # GPT-5
        "openai/gpt-4o-mini",                # GPT-4o mini，便宜快速
        "openai/gpt-4.1",                    # GPT-4.1，质量好
        "deepseek/deepseek-chat",            # DeepSeek，性价比高
        "anthropic/claude-3.5-sonnet",       # Claude 3.5 Sonnet
    ]
    
    print("\n🔍 测试推荐模型...\n")
    
    working_models = []
    for model in recommended_models:
        if test_simple_chat(model):
            working_models.append(model)
        print()
    
    if working_models:
        print("=" * 60)
        print("🎯 可用于分类任务的模型：")
        print("=" * 60)
        for model in working_models:
            print(f"  ✅ {model}")
        
        # 对第一个可用模型进行详细测试
        print(f"\n🧪 对 {working_models[0]} 进行详细分类测试...\n")
        test_model_classification(working_models[0])
    else:
        print("❌ 没有找到可用的模型！请检查API配置。")
        print("💡 提示：运行 'python test_api.py --list' 查看所有可用模型")
