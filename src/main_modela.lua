require 'torch'
require 'math'
require 'os'
local opt = require 'opt'
local loader = require 'text8_loader_modela'
local train_minibatch = require 'train_modela'
local ImageVectors = require 'ImageVectors'

torch.manualSeed(1)
local data = loader.load_data()
local imgFV, imgWords = ImageVectors.getImageVectors()
data.imgFV = imgFV
data.imgWords = imgWords
opt.vocab_size = data.numDistictWords
opt.minibatch_size = 50000
opt.imgdim = data.dim
opt.dim = data.dim
print(opt)
-- local opt = {
--   training_iterations = 200, -- note: the code uses *batches*, not *minibatches*, now.
--   print_every = 1,          -- how many iterations to skip between printing the loss
--   minibatch_size = 1000,
--   vocab_size = data.numDistictWords,
--   dim = data.dim,
--   cuda = 1
-- }

local stime = os.time()
local sclock = os.clock()

model_skipgram, losses_skipgram = train_minibatch(opt, data)

local etime = os.time()
local eclock = os.clock()
print(string.format("walltime = %f sec, cputime = %f sec", etime-stime, eclock-sclock))

-- plot
gnuplot.figure()
gnuplot.plot({'SKIPGRAM',
  torch.range(1, #losses_skipgram), -- x-coordinates
  torch.Tensor(losses_skipgram),    -- y-coordinates
  '-'}
  )
