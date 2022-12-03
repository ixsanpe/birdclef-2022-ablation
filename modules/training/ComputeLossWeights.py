#from ..data import SimpleDataset
from decouple import config
import numpy as np
import pandas as pd
import os
import json

DATA_PATH = config("DATA_PATH")
OUTPUT_DIR = config("OUTPUT_DIR")
birds = ["afrsil1", "akekee", "akepa1", "akiapo", "akikik", "amewig", "aniani", "apapan", "arcter", "barpet", "bcnher", "belkin1", "bkbplo", "bknsti", "bkwpet", "blkfra", "blknod", "bongul", "brant", "brnboo", "brnnod", "brnowl", "brtcur", "bubsan", "buffle", "bulpet", "burpar", "buwtea", "cacgoo1", "calqua", "cangoo", "canvas", "caster1", "categr", "chbsan", "chemun", "chukar", "cintea", "comgal1", "commyn", "compea", "comsan", "comwax", "coopet", "crehon", "dunlin", "elepai", "ercfra", "eurwig", "fragul", "gadwal", "gamqua", "glwgul", "gnwtea", "golphe", "grbher3", "grefri", "gresca", "gryfra", "gwfgoo", "hawama", "hawcoo", "hawcre", "hawgoo", "hawhaw", "hawpet1", "hoomer", "houfin", "houspa", "hudgod", "iiwi", "incter1", "jabwar", "japqua", "kalphe", "kauama", "laugul", "layalb", "lcspet", "leasan", "leater1", "lessca", "lesyel", "lobdow", "lotjae", "madpet", "magpet1", "mallar3", "masboo", "mauala", "maupar", "merlin", "mitpar", "moudov", "norcar", "norhar2", "normoc", "norpin", "norsho", "nutman", "oahama", "omao", "osprey", "pagplo", "palila", "parjae", "pecsan", "peflov", "perfal", "pibgre", "pomjae", "puaioh", "reccar", "redava", "redjun", "redpha1", "refboo", "rempar", "rettro", "ribgul", "rinduc", "rinphe", "rocpig", "rorpar", "rudtur", "ruff", "saffin", "sander", "semplo", "sheowl", "shtsan", "skylar", "snogoo", "sooshe", "sooter1", "sopsku1", "sora", "spodov", "sposan", "towsol", "wantat1", "warwhe1", "wesmea", "wessan", "wetshe", "whfibi", "whiter", "whttro", "wiltur", "yebcar", "yefcan", "zebdov"]

class ComputeLossWeights():
    def __init__(self, beta=0.99,all_birds=birds,metadata=pd.read_csv(f'{DATA_PATH}train_metadata.csv')):
        super().__init__()
        self.beta = beta 
        self.birds = all_birds
        self.metadata = metadata



    def remove_chars(self, s, chars=['[', ']', ' ', '\'']):
        for c in chars:
            s = s.replace(c, '')
        return s

    def forward(self):
        #birds = ["afrsil1", "akekee", "akepa1", "akiapo", "akikik", "amewig", "aniani", "apapan", "arcter", "barpet", "bcnher", "belkin1", "bkbplo", "bknsti", "bkwpet", "blkfra", "blknod", "bongul", "brant", "brnboo", "brnnod", "brnowl", "brtcur", "bubsan", "buffle", "bulpet", "burpar", "buwtea", "cacgoo1", "calqua", "cangoo", "canvas", "caster1", "categr", "chbsan", "chemun", "chukar", "cintea", "comgal1", "commyn", "compea", "comsan", "comwax", "coopet", "crehon", "dunlin", "elepai", "ercfra", "eurwig", "fragul", "gadwal", "gamqua", "glwgul", "gnwtea", "golphe", "grbher3", "grefri", "gresca", "gryfra", "gwfgoo", "hawama", "hawcoo", "hawcre", "hawgoo", "hawhaw", "hawpet1", "hoomer", "houfin", "houspa", "hudgod", "iiwi", "incter1", "jabwar", "japqua", "kalphe", "kauama", "laugul", "layalb", "lcspet", "leasan", "leater1", "lessca", "lesyel", "lobdow", "lotjae", "madpet", "magpet1", "mallar3", "masboo", "mauala", "maupar", "merlin", "mitpar", "moudov", "norcar", "norhar2", "normoc", "norpin", "norsho", "nutman", "oahama", "omao", "osprey", "pagplo", "palila", "parjae", "pecsan", "peflov", "perfal", "pibgre", "pomjae", "puaioh", "reccar", "redava", "redjun", "redpha1", "refboo", "rempar", "rettro", "ribgul", "rinduc", "rinphe", "rocpig", "rorpar", "rudtur", "ruff", "saffin", "sander", "semplo", "sheowl", "shtsan", "skylar", "snogoo", "sooshe", "sooter1", "sopsku1", "sora", "spodov", "sposan", "towsol", "wantat1", "warwhe1", "wesmea", "wessan", "wetshe", "whfibi", "whiter", "whttro", "wiltur", "yebcar", "yefcan", "zebdov"]
        df = self.metadata #pd.read_csv(f'{DATA_PATH}train_metadata.csv')
        primary_labels = df['primary_label'].replace('[', '').replace(']', '')
        primary_labels = pd.Series(primary_labels)
        secondary_labels = df['secondary_labels'].apply(lambda s: self.remove_chars(s).split(','))
        sec_labels = []
        for l in secondary_labels:
            sec_labels.extend(l)
        secondary_labels = pd.Series(secondary_labels)


        labels = np.concatenate([primary_labels, secondary_labels])
        labels = np.delete(labels, np.argwhere(labels == ''))

        counts=[0]*len(self.birds)
        i=0
        for bird in self.birds:
            counts[i]=max(sum(labels==bird),1) #needed for computing weigths later
            i=i+1

        #counts = primary_labels.sum(axis=-1) + secondary_labels.sum(axis=-1)

        #Now, compute weights for the loss as described in https://towardsdatascience.com/handling-class-imbalanced-data-using-a-loss-specifically-made-for-it-6e58fd65ffab
        counts=np.array(counts)
        weights=(1-self.beta)/(1-self.beta**counts)#*10000
        weights1 = weights/max(weights)
        return weights1

#checking the impact of beta
'''weights1=ComputeLossWeights(beta=0.9).forward()
weights2=ComputeLossWeights(beta=0.99).forward()
weights3=ComputeLossWeights(beta=0.999).forward()
weights4=ComputeLossWeights(beta=0.9999).forward()
#print(ComputeLossWeights(beta=0.99).forward())
print(max(weights1)/min(weights1))
print(max(weights2)/min(weights2))
print(max(weights3)/min(weights3))
print(max(weights4)/min(weights4))'''