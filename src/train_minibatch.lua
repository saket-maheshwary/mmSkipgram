--require("cunn")
--require("cutorch")
require 'torch'
require 'math'
require 'os'
require 'nn'
require 'optim'
require 'gnuplot'
-- local mio = require 'matio'
local mio = require 'mattorch'
create_model = require 'create_model_skipgram'
local context = require 'context'
local get_similiar_words = require 'get_similiar_words'
-- print(context)
local trainLogger = optim.Logger('../logs/w2vectrain.log')

local function train_minibatch(opt, data)
    ------------------------------------------------------------------------
    -- create model and loss/grad evaluation function
    --
    local model, criterion, current_vec = create_model.create_model_skipgram(opt)
    local params, grads = model:getParameters()

    -- (re-)initialize weights
--     params:uniform(-0.01, 0.01)
--     if opt.nonlinearity_type == 'requ' then
--         -- need to offset bias for requ/relu/etc s.t. we're at x > 0 (so dz/dx is nonzero)
--         for _, lin in pairs(model:findModules('nn.Linear')) do
--             lin.bias:add(0.5)
--         end
--     end
-- 
    -- return loss, grad
    local feval = function(x)
      if x ~= params then
        params:copy(x)
      end
      grads:zero()

      -- forward
      -- print(minibatch.inputs[1])
      -- print(minibatch.inputs[2])
      -- print(minibatch.targets)
      local outputs = model:forward(minibatch.inputs)
      local loss = criterion:forward(outputs, minibatch.targets)
      -- backward
      local dloss_doutput = criterion:backward(outputs, minibatch.targets)
      model:backward(minibatch.inputs, dloss_doutput)
      -- print(loss)

      return loss, grads
    end

    ------------------------------------------------------------------------
    -- optimization loop
    --
    local losses = {}
    local config = {learningRate = 1e-1,
		    momentum = 0.9,
	            verbose = true, 
		    -- weightDecay = 0.0,
		    -- learningRateDecay = 0.0	
		   }

    local mod = {
	word_vecs = current_vec,
	word2id = data.word2id,
	id2word = data.id2word,
    }
    local word2search = 'cat'
    -- local N =  data.inputs:size()[1]
    local N =  data.totalWords
    for t = 1, opt.training_iterations do

        local stime_e = os.time()
        local sclock_e = os.clock()
        local sum_loss = 0
        local num_batches = 0
        local contextids = nil
        local words = nil
        local targets = nil
        local state = {}
        -- for i = 1, N, opt.minibatch_size do
        for i = data.contextSize/2 + 1, N-(data.contextSize/2), opt.minibatch_size do
	  
            local stime = os.time()
            local sclock = os.clock()
          local mb_size = math.min(N-i-data.contextSize/2, opt.minibatch_size)
          --inputs = nil
          --targets = nil
          --inputs, targets = context.get_context(data, i, i+mb_size-1)
          -- print(inputs[1]:size())
          contextids, words, targets = context.get_context(data, i, i+mb_size-1)
          -- print(inputs[2]:size())
          -- print(targets)
	  if opt.cuda == 1 then
	  	-- require("cunn")
		-- require("cutorch")
		contextids = contextids:cuda()
		words = words:cuda()
		targets = targets:cuda()
	  end
          minibatch = {
              inputs = {contextids, words},
              targets = targets
          }
          --print(minibatch.inputs)
           -- minibatch = {
           --    inputs = data.inputs[{{i,i+mb_size-1},{}}],
           --    targets = data.targets[{{i,i+mb_size-1}}]
           -- }  

	   -- print(params)
           local _, loss = optim.adagrad(feval, params, config, state)
          -- print(loss:size())
	  num_batches = num_batches + 1 
	  sum_loss = sum_loss + loss[1]
	  
	  local etime = os.time()
	  local eclock = os.clock()
--           if (i - data.contextSize/2+1) % opt.print_every == 0 then
--               print(string.format("epoch %d iteration %4d, loss = %6.6f, walltime = %f, cputime = %f",t, i, loss[1],etime-stime,eclock-sclock))
--           end
        end  -- end of i loop

	if opt.cuda == 1 then
		contextids = nil --contextids:float()
		words = nil --words:float()
		targets = nil --targets:float()
	end
        losses[#losses + 1] = sum_loss -- append the new loss
	sum_loss = sum_loss / num_batches
	trainLogger:add{['Training Loss'] = sum_loss}
	trainLogger:style{['Training Loss'] = '-'}
	trainLogger:plot()
	
	
	-- if t%10 == 0 then	
		local fname = string.format("../output/models/w2vec_m_%d.mat",t)
	    	mio.save(fname,current_vec.weight:clone():double())
		mod.word_vecs = current_vec
	    	r = get_similiar_words.get_sim_words(mod,word2search, opt.knn)
     -- end

    local etime_e = os.time()
    local eclock_e = os.clock()
	print(string.format("Epoch: %4d, loss = %6.6f, walltime=%f, cputime=%f", t, sum_loss, etime_e-stime_e, eclock_e-sclock_e))
    end

    mod.word_vecs = current_vec    
    r = get_similiar_words.get_sim_words(mod,word2search, opt.knn)
    return model, losses
end

return train_minibatch
