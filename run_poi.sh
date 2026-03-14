PYTHONPATH=src python src/sensei-chat.py \
    --technology convnavi \
    --chatbot http://127.0.0.1:8000/query \
    --user examples/profiles/poi-search \
    --personality ./personalities_car/ \
    --save_folder results/poi-search \
    --generator_llm "gpt-4o-mini" \
    --judge_llm "gpt-5-mini" \
    --population_size 2 \
    --max_time "00:01:00"
