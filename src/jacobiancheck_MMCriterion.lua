-- require 'requ'
-- require 'MMCriterion'
require 'TripletCriterion'
grad = require 'autograd'
local gradcheck = require 'autograd.gradcheck'

local function jacobian_wrt_input(module, x, eps)
  
  local z = module:forward(x)

  grad_est = {}
  for k = 1, #x do              -- number of inputs
      grad_est[#grad_est + 1] = torch.zeros(x[k]:size()):clone()
      for l = 1, x[k]:size()[1] do
	for i = 1, x[k]:size()[2] do -- dimensions
            x[k][{{l}, {i}}] = x[k][{{l}, {i}}] + eps
            local z_plus = module:forward(x)
          --  module.output = 0
            x[k][{{l}, {i}}] = x[k][{{l}, {i}}] - 2 * eps
            local z_minus = module:forward(x)
          --  module.output = 0
           x[k][{{l}, {i}}] = x[k][{{l}, {i}}] + eps -- important ! restore vector
--            x[k][{{}, {i}}] = x[k][{{}, {i}}] - eps -- important ! restore vector
            -- print(z_plus - z_minus)
            grad_est[k][{{l},{i}}] = (z_plus - z_minus) / (2 * eps)
        end
    end
  end

  local grad = module:backward(x)

  for k = 1, #x do
      local d = (grad[k] - grad_est[k]):abs()
      print(torch.max(d), torch.min(d), torch.mean(d))
  end

  
  -------------------------------------------------------------------------

--   local z = module:forward(x):clone()
--   local jac = torch.DoubleTensor(z:size(1), x:size(1))
--   
--   -- get true Jacobian, ROW BY ROW
--   local one_hot = torch.zeros(z:size())
--   for i = 1, z:size(1) do
--     one_hot[i] = 1
--     jac[i]:copy(module:backward(x, one_hot))
--     one_hot[i] = 0
--   end
--   
--   -- compute finite-differences Jacobian, COLUMN BY COLUMN
--   local jac_est = torch.DoubleTensor(z:size(1), x:size(1))
--   for i = 1, x:size(1) do
--     -- TODO: modify this to perform a two-sided estimate. Remember to do this carefully, because 
--     --       nn modules reuse their output buffer across different calls to forward.
--     -- ONE-sided estimate
--     x[i] = x[i] + eps
--     local z_offset = module:forward(x)
--     x[i] = x[i] - eps
--     jac_est[{{},i}]:copy(z_offset):add(-1, z):div(eps)
--   end
-- 
--   -- computes (symmetric) relative error of gradient
--   local abs_diff = (jac - jac_est):abs()
--   return jac, jac_est, torch.mean(abs_diff), torch.min(abs_diff), torch.max(abs_diff)
end

---------------------------------------------------------
-- test our layer in isolation
--
torch.manualSeed(1)
-- local requ = nn.ReQU()

-- local x = torch.randn(10) -- random input to layer
-- print(x)
-- print(jacobian_wrt_input(requ, x, 1e-6))

local n = 10
local data = {}
local dim = 4
local vocab_size = 100 
local wordFVocab = torch.randn(vocab_size, dim)
local imgFVocab = torch.randn(vocab_size, dim)
local wordF = torch.randn(n, dim)
local imgF = torch.randn(n, dim)
local negF = torch.Tensor(n, dim)
local word_id = torch.Tensor(n)
word_id:random(vocab_size)
local posimage_id = word_id:clone()
for i = 1, n do
    wordF[{{i}, {}}] = wordFVocab[{{word_id[i]}, {}}]
    imgF[{{i}, {}}] = imgFVocab[{{posimage_id[i]}, {}}]
    local negid = torch.random(vocab_size)
    while negid == posimage_id[i] do
        negid = torch.random(vocab_size)
    end
    negF[{{i}, {}}] = imgFVocab[{{negid}, {}}]
end
print(wordF)
print(imgF)
print(negF)
data.inputs = {wordF, imgF, negF}
data.targets = torch.rand(n):mul(3):add(1):floor()  -- random integers from {1,2,3}
print(data)

-- print(jacobian_wrt_input(nn.MMCriterion(), data.inputs, 1e-6))
print(jacobian_wrt_input(nn.TripletCriterion(), data.inputs, 1e-6))
--

-- local params = {
--     a = torch.randn(n, dim),
--     p = torch.randn(n, dim),
--     n = torch.randn(n, dim)
-- }
-- 
-- -- local inputs = {a=wordF, p=imgF, n=negF}
-- local inputs = {a, p, n}
-- 
-- local tripletloss = function(params, inputs)
-- 
--     local margin = 1.0
-- 
--     -- Normalization
-- 
--     --  local ad = torch.norm(params.a, 2, 2)
--     -- local pd = torch.norm(params.p, 2, 2)
--     -- local nd = torch.norm(params.n, 2, 2)
--     -- params.a = params.a:cdiv(ad:repeatTensor(1, params.a:size(2)))
--     -- params.p = params.p:cdiv(pd:repeatTensor(1, params.p:size(2)))
--     -- params.n = params.n:cdiv(nd:repeatTensor(1, params.n:size(2)))
-- 
--     -- Loss
-- 
--     -- local loss = params.n:cmul(params.a):add(params.p:cmul(params.a):mul(-1.0)):sum(2.0):add(margin):cmax(0.0):sum(1.0)  
--     local loss = margin - torch.dot(params.a, params.p) + torch.dot(params.a, params.n) 
--     return loss
-- end
-- 
-- 
-- -- loss = tripletloss(params, input)
-- grads, loss = grad(tripletloss)(params, inputs)
-- print('grads: ', grads, 'loss:', loss)
-- 
-- -- tester = torch.Tester()
-- -- tester:assert(gradcheck(tripletloss, {a=wordF, p=imgF, n=negF}, a), 'incorrect gradients on a')
-- -- tester:assert(gradcheck(tripletloss, {a=wordF, p=imgF, n=negF}, p), 'incorrect gradients on p')
-- -- tester:assert(gradcheck(tripletloss, {a=wordF, p=imgF, n=negF}, n), 'incorrect gradients on n')
