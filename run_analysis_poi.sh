
subfolder="2026-02-09-00-10-03"

PYTHONPATH=src python src/sensei-check.py \
        --rules ./examples/rules/poi-search/ \
        --conversations ./results/poi-search/poi_search_test_custom/$subfolder \
        --verbose \
        --dump ./results/poi-search/poi_search_test_custom/$subfolder/stats.csv