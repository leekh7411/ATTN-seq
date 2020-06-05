from attnseq import *
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def main(params):
    benchmark = 'B'
    inp_lang  = 'p{}'.format(benchmark)
    out_lang  = 'r{}'.format(benchmark)
    hidden_size = 300
    print_every = 50
    num_iters = 50000
    learning_rate = 0.01
    
    
    print("")
    print("[DATASET]")
    input_lang_train, output_lang_train, pairs_train = prepareData(inp_lang, out_lang, False, True)
    input_lang_test, output_lang_test, pairs_test = prepareData(inp_lang, out_lang, False, False)
    print(len(pairs_train))
    print(len(pairs_test))
        
    print("")
    print("[MODELS]")
    encoder1      = EncoderRNN(input_lang_train.n_words, hidden_size).to(device)
    attn_decoder1 = AttnDecoderRNN(hidden_size, output_lang_train.n_words, dropout_p=0.1).to(device)
    print(encoder1)
    print(attn_decoder1)
    
    print("")
    print("[TRAIN]")
    trainIters(input_lang_train,
               output_lang_train,
               pairs_train, 
               encoder1,
               attn_decoder1,
               num_iters,
               print_every=print_every,
               learning_rate=learning_rate)
    
    print("")
    print("[EVALUATION]")
    evaluateRandomly(encoder1, 
                     attn_decoder1,
                     input_lang_test,
                     output_lang_test,
                     pairs_test)
    
if __name__ == "__main__":
    main({})