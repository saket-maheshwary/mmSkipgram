require 'torch'
require 'math'
require 'os'

local opt = require 'opt'
local loader = require 'text8_loader'
local train_minibatch = require 'train_minibatch'

torch.manualSeed(1)
local data = loader.load_data()
opt.vocab_size = data.numDistictWords
opt.dim = data.dim
opt.minibatch_size = 50000
-- local opt = {
--   training_iterations = 200, -- note: the code uses *batches*, not *minibatches*, now.
--   print_every = 1,          -- how many iterations to skip between printing the loss
--   minibatch_size = 100000,
--   vocab_size = data.numDistictWords,
--   dim = 300,
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
