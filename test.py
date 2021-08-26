from dataloader import DataLoader, DataFilter
from eval import EvalClass
from bert import load_final_model
from icecream import ic
from transformers import TrainingArguments, Trainer
from transformers import AutoTokenizer
from datasets import load_dataset
from utils import set_seed
from config import cfg

import numpy as np
import pandas as pd

label_list = cfg['LABEL_LIST']

def test():
    model = load_final_model()
    dataset = DataLoader(cfg, mode = '5-cross-inference')
    args =  TrainingArguments(
            f"model-5-cross-{cfg['TASK']}-final",
            evaluation_strategy = "epoch",
            per_device_train_batch_size=cfg['BATCH_SIZE'],
            per_device_eval_batch_size=cfg['BATCH_SIZE'],
            num_train_epochs=3,
            learning_rate = cfg['LR'],
            weight_decay=0.01,
        )
    evalClass = EvalClass() 
    trainer = Trainer(model, args, train_dataset = dataset.train, eval_dataset = dataset.train,compute_metrics=evalClass.compute_metrics)
    print(trainer.evaluate())
# Test the output model

def inference_simple_text(text):
    import torch
    tokenizer = AutoTokenizer.from_pretrained(cfg['MODEL_NAME'])
    tokenized_inputs = tokenizer(text,padding="max_length", truncation=True)
    model = load_final_model()
    device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
    input_ids = torch.tensor([tokenized_inputs['input_ids']]).to(device)
    attention_mask = torch.tensor([tokenized_inputs['attention_mask']]).to(device)
    output = model(input_ids = input_ids, attention_mask = attention_mask)
    output = torch.sigmoid(output.logits).cpu().detach().numpy().tolist()
    print(output)
    return output

if __name__=='__main__':
    inference_simple_text('I offer the following resolutions: Resolved. That so much of the anuoal message of the President of tbe united States to the two houses of Congress at the present session. together with the acncipancing documnats. as relates to finanee and taxation. to the receipts into the Tresury. to deficiencies in the revenue. the revision of the tariff aid internal revcane laws. to the establishing ofuints. to the public debt and public credit and to the ways and nians of supporting and meeting the liabilities of the Government. be refored to the Committee on Ways and Means. ERselved. That so tniel of said message and documents as relates to the necesnary appropriation for earryitng sl the Goverineat in its several departments. to delicieneics in aplcopriatintts. to postal telegraphy. to special appropriation for meeting te award to G reat]ritain agaiisttheUnited States to special appropriation to the government of the TeritOy of the District of Columbia. and to special appropriation for ocean steam service. be referred to the Committee on Approlpations. Resleed. Titat so mtteh of said message sad documents as rolates to baults and banlidt cud currency. be referrned to tlte Ceomittee on Bankcing and Corretney. )Resolved. Fit so mact of sauittmessago and documents as re lates to coamerco and nacigatioct ho retered to tne Committee oct Cotutneree. eolced. Flint so tuc of said message and docents as relates to the settling of slenies of imeigrants on te public lands. to the sutvev and preseralien of Governmttect reservtiont. itcltding te park of te Yellowstone. ad to the petite docitain. be referral to te Coctnittee ot tite Poblie lands. feselved. That so ncet of said message and documents as relates to the PoseOithee Depaatcnt. to land. coast. postal detositries. acid ocean mail servie be refrred ti tle Co t mtcitte oct taes ostOflieacitu PostRoads. lfseclcdcf. that so mudl. c said isessage aitt docuetits as relates to courts and the iudiciry. to cidditi. not eeotcnto to tlie Cocnstitestiett to the repeal or nedtfccctitt ct tkrutpt st. tct civil rightts." to confliot of ccnthority in thi coirts itt Utahc. e. tt ticiteva awcrd. ani to the Departmect of Jastiee. be referred to the Co tttitree f. tCe J rdiciary. oceleed. That so ntei of said tessage ad docemeuts as relates to the publie expend"titres b r e lrrd to the Comifte onllie E eedicur Resolved. Thlat so much of said mnessage ad documents as relates to agrieoltre antice Dopartatnt of Agricttre be referred to lbs Comnmitteo on Agricallieeced. Thlat so much( of said essage and documents as relates to ths amy of te United States. tlto aenmanent for seacoast defenses. and te proootioe in tei staff ceos of tile Arny. bcreferred to tlh Coepimittee on Jilitccry A firs. Resoled. That so cuch of saih message and documents as relates to the pavy of etieit Sttes be r eferred to th e Comtcnitte on NavEl Axftuirs. Rsolve . That so iuclh of said tuessage atd doetcuents and tle aecompanying rresplcdec. cis8 relcates tic fortigit alflirs. to treaties witht foreign gtovecicoents atic to ht. seiaro of the Tirgiuitt cmnd all questions greosing out of the sae. be retfirrc dr to the Ctntittee ot Foreign AAirs. Resolved. That so mc of said message and documents as relates to the managemcnt of Iiia adhfirs cbe referred to the Cocuiittee on Itdian Aflairs. Rcsoefcd. That so nouch of said message ald documents as relates to the Pacific lisilrodeie bie referred to tice Cocittee on te Pacific Raitoaid. kesoleed. That so cuc of sal nessage and documents as relates to citis against htscitivertcntet. nit ineinding clainc s growing o ut of a ty war in wtich the United States hs been engaged. e rferrd to the Committee on Clains. lfe elved. That so mict of said nessago and documents as relates to manfautores Ic s be re ferred to ti Conomtetta on Aauafactures. Resolved. That so much of said iessag cnd documents as relates te private landclaims be referred to the Committee octPiivate Land Claicas. Resolve sd. Tiat so mcei of said message an docments as relates to mines and mictinig be t eferred to the Ceomittee on oies acd Mining. Resolved. That so iuch of said message and documents as relates to coinage. cigicts. aid tessures be referred to the Comuithee on Coinage. Wights acd at[ecaslres. Resolecf. That so much of said message and documents as relates to public hmililngs and groutns be referred he the Cottittee on tiblic iildiccgs and Grounds. Resoleed. That so mli of said coessage and doeumients as relates to he inapen t et O fts polt buildings be referred to h e Committee on Expendihtes on Poliic ciidiigs. l esolved. Tt. so much of said message and documents as relates to the Territories i the United Stcchs be referred t. ttic Comittee on the Territories. Cemictee Oct itnvalidl tcnsiocs. Resolved. That so iucel of said cesago acid documents as relates to patets and the PattOulice ho referred to cte Coieichee ott Patents. Resolved. That so much of said message and documents as relates to exponditures in connection with the Treasury Decartmont be referred to the Committee cta EX1ctilibires il ts Tretoury ecricet. Resolved. Thit so mttieh of said iceCSSage and documents as relates to expendi. tures in connection with the State Department be referred to the Committee on Expenditcres in the State Department. fResolsed. That so incit of said itessage and documents as relates to tic expendi. tures it ceonnection with the War Department be referred to the Committee onExpenditures in tite War Departcent. Resolved. That so much of said itessage and documents as relates to the expendi. tucres in connection with the Navy Deprtment be referred to the Committee on Expenditures in the C avy Dcartinent. Resoled. Tha so much of sad tessage and documents as relates to expenditures in connection with tie PostDico Department be referred to the Committee on Expenditres in the PostGtluts Deciartmenct. Resolved. That so Much of said ucessageand documents as relates to expenditures i connection with tice Department if the Interior be referred to the Conmittee on Expenditireos in the Interr )teirtmect. Ressolved. That ot much of said ieesago and documents as relates to the militia be referred to.the Comittee oil the Militia. Resolved. That so much of sail mssage and documents as relates to tle Territory of the District of Columbia be referred to tho Committee for the District of Columbia. Resol ecd. That so much of said message and docunents as relates to education acid labor. to a national niversity in the District of Columbia. and to the Btaeau of Eiducation. be referred to the (Ioet ittee on Education and Labor. Resolved. That so m ucl of said itessage and doeucnts as relates to eiilservice refor be referred t tie Select Committee on the feorganization of the Civil Service of lic United States. Resolved. That so much of said message and documents as relates to railways and ianals. and all cicstiocns growing out of tice subject of cheap trausportation Ice refirreid t Cte Cumuittce on Railways and cattals. lesolved. That so much of said icesoage and documents as relates to claims growing outer aly war in which the United States has been engaged. be referred to the Comiciittee oi War Claims. JResolved. ihat so much of said messae atnd documents as relates to the proposed centeocial ceeliration. to a naiocal eletss iu 1875. and to the late Vienia E xposition. be referred to a select committee of thirteen members. to be appointed by the Speaker.')