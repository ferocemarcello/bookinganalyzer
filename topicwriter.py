import helper
import fasttext

def do(originfile):
    keywords=helper.getKeywords(originfile)
    print(len(keywords))
    '''model = fasttext.train_supervised(input="resources/cooking.train")
    model.save_model("model_cooking.bin")
    print(model.predict("Which baking dish is best to bake a banana bread ?"))
    print(model.predict("Why not put knives in the dishwasher?"))
    print(model.predict("Why not put knives in the dishwasher?", k=5))'''
    model=fasttext.load_model('resources/amazon_review_full.bin')
    print(model.predict("Which baking dish is best to bake a banana bread ?",k=5))
    print("cacca")