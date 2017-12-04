require 'torch'
local mio = require 'matio'
local json = require 'dkjson'


local loader = {}

function loader.load_data()
  -- load
  local data = {}
  data.inputs = {}
  data.targets = {}
  data.targets_by_name = {}
  data.vocab = {}
  data.numDistictWords = 0
  data.totalWords = 0
  data.word2id = {}
  data.id2word = {}
  data.contextSize = 4
  data.negImgSize = 20
  data.text = {}
  data.corpustemp = {}
  data.imgFV = {}
  data.imgWords = {}
  data.dim = 300
  local w2idfname = "../output/vocab/w2id.t7"
  local w2idfname_json = "../output/vocab/w2id.json"
  local id2wfname = "../output/vocab/id2w.t7"
  local id2wfname_json = "../output/vocab/id2w.json"
  -- corpusfile = "../data/text8_proc"
  corpusfile = "../data/text8_proc_lemmatized"
  
  local f = io.open(corpusfile, "r")
  -- print(f)
    local block = 8487590
    data.corpustemp =  f:read("*all")
    data.corpus = string.sub(data.corpustemp, 1, block)
   -- data.corpus = f:read("*all")
   print('Read corpus ', corpusfile, '\n')
 -- print (type(data.corpus))
  for word in string.gmatch(data.corpus, "%S+") do
    -- print(word)
    if data.vocab[word] == nil then
        data.vocab[word] = 1
        data.numDistictWords = data.numDistictWords + 1
        data.id2word[data.numDistictWords] = word
        data.word2id[word] = data.numDistictWords
    else
        data.vocab[word] = data.vocab[word] + 1
    end
    data.text[#data.text + 1] = word
    data.totalWords = data.totalWords + 1
  end
  io.close(f)
  
  torch.save(w2idfname,data.word2id)
  torch.save(id2wfname,data.id2word)
  w2idf = io.open(w2idfname_json, 'w')
  w2idf:write(json.encode(data.word2id,{indent = true} ))
  w2idf:close()
  id2wf = io.open(id2wfname_json, 'w')
  id2wf:write(json.encode(data.id2word, {indent = true}))
  id2wf:close()
  -- print('WORD2ID:')
  -- print(data.word2id)
  -- print('ID2WORD:')
  -- print(data.id2word)
  print('Number of distinct words: ', data.numDistictWords)
  print('Total number of words: ', data.totalWords)

  return data
end

return loader

