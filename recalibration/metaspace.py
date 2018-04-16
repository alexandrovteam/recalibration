from elasticsearch import Elasticsearch
from pyMSpec.instrument import FTICR
import pandas as pd

def get_mz(ion, charge):
    f = FTICR(resolving_power = 300000, at_mz=400)
    return f.get_principal_peak(ion, charge)

def get_reference_mzs(polarity, tissue_type="", source="", database='HMDB', fdr = 0.1, config={}, min_count=0):
    esclient = Elasticsearch("http://{}:{}@metaspace2020.eu:9210".format(config['user'], config['password']), timeout=100)
    assert polarity in ['Positive', 'Negative']
    charge = 1
    if polarity == 'Negative':
        charge = -1
    filter =  [
            {"term": {"ds_meta.MS_Analysis.Polarity": polarity}},
            {"range": {"fdr": {"lte": fdr}}},
            {"term": {"db_name": database}},

        ]

    if tissue_type != "":
        filter += [{"term": {"ds_meta.Sample_Information.Organism_Part": tissue_type}},]

    if source != "":
        filter += [{"term": {"ds_meta.MS_Analysis.Ionisation_Source": source}},]
    print(filter)
    response = esclient.search(
        index='sm-*',
        body={
            "query": {
            "bool":{
                "filter": filter
                },
            },
            "aggs": {
                "sf_adduct": {
                    "terms": {
                        "field": "sf_adduct",
                        "size": 5000
                    }
                },
                "ds_id":{
                    "terms":{
                        "field": "ds_id",
                        "size": 5000
                    }
                },
                "mzs":{
                "terms":{
                        "field": "mz",
                        "size": 5000
                    }
                },
            }
        }
    )
    count = pd.Series(
            {bucket["key"]: bucket["doc_count"] for bucket in response['aggregations']['sf_adduct']['buckets']}
        ).sort_values(ascending=False)*100./len(response['aggregations']["ds_id"]["buckets"])
    mzs = pd.Series({ion: get_mz(ion, charge) for ion in count.index}).loc[count.index]
    mzs.rename('mz', inplace=True)
    count.rename('count', inplace=True)
    count += min_count
    return mzs, count

