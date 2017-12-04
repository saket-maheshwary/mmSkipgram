require 'torch'

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
  data.text = {}

  local f = io.open("text8_proc", "r")
  print(f)

  data.corpus =  f:read("*all")
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

  print('Number of distinct words: ', data.numDistictWords)
  print('Total number of words: ', data.totalWords)

  return data
end

return loader

