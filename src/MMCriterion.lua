require 'math'
require 'nn'


local MMCriterion, parent = torch.class('nn.MMCriterion', 'nn.Criterion')

function MMCriterion:__init(margin)
    parent.__init(self)
    self.margin = margin or 1
    self.gradInput = {}
end

function MMCriterion:updateOutput(input, target)

    -- print(input)
    local a = torch.squeeze(input[1],2)  -- anchor
    local p = input[2]  -- positive
    local n = input[3]  -- negative
    local num_mb = n:size(1)
    local num_neg = n:size(2)
    local num_dim = n:size(3)
    
    --print(a:size())
    --print(p:size())
    --print(n:size())
    -- Normalization
    local adiv = torch.norm(a, 2, 2)
    local pdiv = torch.norm(p, 2, 2) 
    --a = a:cdiv(adiv:repeatTensor(1, a:size(2)))
    --p = p:cdiv(pdiv:repeatTensor(1, p:size(2)))
    
--    local nchk, _ = p:ne(p):max(2)
--    nchk = torch.squeeze(nchk)
    local ncount = 0
--     for i = 1, num_mb do
--         for k = 1, num_neg do
-- 	 	n[{{i}, {k}, {}}] = n[{{i}, {k}, {}}] / torch.norm(n[{{i}, {k}, {}}], 2)
--         end
--     end
    

    self.loss = 0
    for i = 1, num_mb do
       	if pdiv[i][1] ~= 0 then
		for k = 1, num_neg do
       		self.loss = self.loss + math.max(0, self.margin - torch.dot(a[{{i},{}}], p[{{i},{}}])/(pdiv[i][1]*adiv[i][1]) +  torch.dot(a[{{i},{}}], n[{{i},{k},{}}])/(adiv[i][1]*torch.norm(n[{{i}, {k}, {}}], 2)))
       		end
	else
		ncount = ncount + 1
	end
    end

--     print(self.loss)
--     print(ncount)
--     print(num_mb)
    if ncount < num_mb then
--	print("Inside if")
    	self.loss = self.loss / (num_mb - ncount)
    end
    return self.loss
end


function MMCriterion:updateGradInput(input, target)

    local a = torch.squeeze(input[1],2)  -- anchor
    local p = input[2]  -- positive
    local n = input[3] -- negative

    local num_mb = n:size(1)
    local num_neg = n:size(2)
    local num_dim = n:size(3)

    -- Normalization

    local adiv = torch.norm(a, 2, 2)
    local pdiv = torch.norm(p, 2, 2)
   -- a = a:cdiv(adiv:repeatTensor(1, a:size(2)))
   -- p = p:cdiv(pdiv:repeatTensor(1, p:size(2)))

  --  local nchk, _ = p:ne(p):max(2)
  --  nchk = torch.squeeze(nchk)
    local ncount = 0
    
--     for i = 1, num_mb do
--         for k = 1, num_neg do
-- 	 	n[{{i}, {k}, {}}] = n[{{i}, {k}, {}}] / torch.norm(n[{{i}, {k}, {}}], 2)
--         end
--     end

    self.gradInput = {torch.zeros(a:size()):typeAs(input[1]), torch.zeros(p:size()):typeAs(input[2]), torch.zeros(n:size()):typeAs(input[3]) }

     for i = 1, num_mb do
        if pdiv[i][1] ~= 0 then 
	for k = 1, num_neg do
             local l = self.margin - torch.dot(a[{{i},{}}], p[{{i},{}}])/(adiv[i][1]*pdiv[i][1]) +  torch.dot(a[{{i},{}}], n[{{i},{k},{}}])/(adiv[i][1]* torch.norm(n[{{i}, {k}, {}}], 2))
             local ind = 0
             if l > 0 then
                 ind = 1
             end
	     local ndiv = torch.norm(n[{{i}, {k}, {}}], 2)
             self.gradInput[1][{{i}, {}}] = self.gradInput[1][{{i}, {}}] + (n[{{i}, {k}, {}}]:resizeAs(self.gradInput[1][{{i}, {}}])/(adiv[i][1]*ndiv) - p[{{i}, {}}]/(adiv[i][1]*pdiv[i][1])) * ind
             self.gradInput[2][{{i}, {}}] = self.gradInput[2][{{i}, {}}] - a[{{i}, {}}]/(adiv[i][1]*pdiv[i][1]) * ind
             self.gradInput[3][{{i}, {k}, {}}] = self.gradInput[3][{{i}, {k}, {}}] + a[{{i}, {}}]:resizeAs(self.gradInput[3][{{i}, {k}, {}}])/(adiv[i][1]*ndiv) * ind
         end
	end
     end
    
     --print(self.gradInput[1]:size())
     --print(self.gradInput[2]:size())
     --print(self.gradInput[3]:size())
    
    return self.gradInput
end
