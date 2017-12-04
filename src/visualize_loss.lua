require 'torch'
require 'nn'
require 'cunn'
local loader = require 'text8_loader'
local get_similiar_words = require 'get_similiar_words'

local data = loader.load_data()


torch.manualSeed(1)
local t = 60
local fname = string.format("../output/models/w2vec%d.t7",t)
local current_vec = torch.load(fname)

mod = {
word_vecs = current_vec,
word2id = data.word2id,
id2word = data.id2word,
}

-- print(mod.word2id)

local word2search = 'western'
-- local word2search = 'appraisal'
-- local word2search = 'education'

r = get_similiar_words.get_sim_words(mod,word2search,100)
return model, losses
