require 'torch'
require 'math'
require 'nn'
require 'optim'
require 'gnuplot'
require 'requ'
create_model = require 'create_model_skipgram'
local context = require 'context'
print(context)

local function train_minibatch(opt, data)
    ------------------------------------------------------------------------
    -- create model and loss/grad evaluation function
    --
    local model, criterion = create_model.create_model_skipgram(opt)
    local params, grads = model:getParameters()

    -- (re-)initialize weights
    params:uniform(-0.01, 0.01)
    if opt.nonlinearity_type == 'requ' then
        -- need to offset bias for requ/relu/etc s.t. we're at x > 0 (so dz/dx is nonzero)
        for _, lin in pairs(model:findModules('nn.Linear')) do
            lin.bias:add(0.5)
        end
    end

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

      return loss, grads
    end

    ------------------------------------------------------------------------
    -- optimization loop
    --
    local losses = {}
    local optim_state = {learningRate = 1e-1}

    -- local N =  data.inputs:size()[1]
    local N =  data.totalWords
    for t = 1, opt.training_iterations do

        local sum_loss = 0
        -- for i = 1, N, opt.minibatch_size do
        for i = data.contextSize/2 + 1, N, opt.minibatch_size do

          local mb_size = math.min(N-i, opt.minibatch_size)
          local inputs = nil
          local targets = nil
          --inputs, targets = context.get_context(data, i, i+mb_size-1)
          -- print(inputs[1]:size())
          contextids, words, targets = context.get_context(data, i, i+mb_size-1)
          -- print(inputs[2]:size())
          -- print(targets)
          minibatch = {
              inputs = {contextids, words},
              targets = targets
          }
          --print(minibatch.inputs)
           -- minibatch = {
           --    inputs = data.inputs[{{i,i+mb_size-1},{}}],
           --    targets = data.targets[{{i,i+mb_size-1}}]
           -- }  

          local _, loss = optim.adagrad(feval, params, optim_state)
          sum_loss = sum_loss + loss[1]

          if i % opt.print_every == 0 then
              print(string.format("iteration %4d, loss = %6.6f", i, loss[1]))
          end
        end
        losses[#losses + 1] = sum_loss -- append the new loss
        print(string.format("Epoch: %4d, loss = %6.6f", t, sum_loss))
    end


    return model, losses
end

return train_minibatch

