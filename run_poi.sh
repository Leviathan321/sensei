PYTHONPATH=src python src/sensei-chat.py \
    --technology convnavi \
    --chatbot http://127.0.0.1:8000/query \
    --user examples/profiles/poi-search \
    --extract results/poi-search
