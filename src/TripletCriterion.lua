require 'math'
require 'nn'


local TripletCriterion, parent = torch.class('nn.TripletCriterion', 'nn.Criterion')

function TripletCriterion:__init(margin)
    parent.__init(self)
    self.margin = margin or 1
    self.gradInput = {}
end

function TripletCriterion:updateOutput(input)

    local a = input[1]  -- anchor
    local p = input[2]  -- positive
    local n = input[3]  -- negative

    -- Normalization

    local adiv = torch.norm(a, 2, 2)
    local pdiv = torch.norm(p, 2, 2)
    local ndiv = torch.norm(n, 2, 2)
    a = a:cdiv(adiv:repeatTensor(1, a:size(2)))
    p = p:cdiv(pdiv:repeatTensor(1, p:size(2)))
    n = n:cdiv(ndiv:repeatTensor(1, n:size(2)))

    -- Loss

--    loss = n:cmul(a):add(p:cmul(a):mul(-1)):sum(2):add(self.margin):cmax(0):sum(1)  

    loss = n:cmul(a):sum(2):add(p:cmul(a):sum(2):mul(-1)):add(self.margin):cmax(0)
    loss = loss:sum(1)/a:size()[1]
    -- print(self.loss)
    return loss
end


function TripletCriterion:updateGradInput(input)

    print('updateGradInput called')
    local a = input[1]  -- anchor
    local p = input[2]  -- positive
    local n = input[3]  -- negative

    -- Normalization

    local adiv = torch.norm(a, 2, 2)
    local pdiv = torch.norm(p, 2, 2)
    local ndiv = torch.norm(n, 2, 2)
    a = a:cdiv(adiv:repeatTensor(1, a:size(2)))
    p = p:cdiv(pdiv:repeatTensor(1, p:size(2)))
    n = n:cdiv(ndiv:repeatTensor(1, n:size(2)))

    local aa = a:clone()
    local pp = p:clone()
    local nn = n:clone()
    
    -- Gradients

    gradInput = {torch.zeros(a:size()), torch.zeros(p:size()), torch.zeros(n:size()) }

--    local ind = torch.gt(n:cmul(a):add(p:cmul(a):mul(-1)):sum(2):add(self.margin) , torch.zeros(a:size()[1], 1)):double()

    local ind = torch.gt(n:cmul(a):sum(2):add(p:cmul(a):mul(-1):sum(2)):add(self.margin) , torch.zeros(a:size()[1], 1)):double()
   --  print(input[1]:size())
    gradInput[1] = (nn - pp):cmul(ind:repeatTensor(1, a:size()[2]))
    gradInput[3] = aa:cmul(ind:repeatTensor(1, a:size()[2]))
    gradInput[2] = gradInput[3]:mul(-1)

    -- print(self.gradInput[1]:size())
    -- print(self.gradInput[2]:size())
    -- print(self.gradInput[3]:size())
    print(gradInput)
    
    return gradInput
end
