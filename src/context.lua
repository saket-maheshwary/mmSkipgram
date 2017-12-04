local context = {}

function context.get_context(data, mb_start, mb_end)

  local inputs = {}
  local contextids = {}
  local wordid = {}
  local contextid_minibatch = {}
  local wordid_minibatch = {}
  local targets = {}
  local labels = {}
  for i = mb_start, mb_end do
        if i > data.contextSize / 2 then
            contextids = {}
            for j = i - data.contextSize / 2, i + data.contextSize / 2 do
                if j ~= i then
                    contextids[#contextids + 1] = data.word2id[data.text[j]]
                    targets[#targets + 1] = 1
                end
            end
            --print(targets)
            wordid = data.word2id[data.text[i]]
            k = 0
            while k < data.contextSize do
                found = false
                c = torch.random(data.numDistictWords)
                for l = 1, #contextids do
                    if contextids[l] == c then
                        found = true
                        break
                    end
                end
                if found == false then
                      contextids[#contextids + 1] = c
                      targets[#targets + 1] = 0
                      k = k+1
                end
           end
           --contextids = torch.Tensor(contextids)
            --print(wordid)
            contextid_minibatch[#contextid_minibatch + 1] = contextids
            wordid_minibatch[#wordid_minibatch + 1] = wordid
            --inputs[#inputs + 1] = {contextids, word2id}
            labels[#labels + 1] = targets
            wordid = {}
            contextids = {}
            targets = {}
            -- collectgarbage()
        end
  end
  local cid = torch.Tensor(mb_end-mb_start+1, 2*data.contextSize)
  --local cid = torch.Tensor(2*data.contextSize, mb_end-mb_start+1)
  local wid = torch.Tensor(mb_end-mb_start+1, 1)
  --local lab = torch.IntTensor(2*data.contextSize,mb_end-mb_start+1)
  local lab = torch.Tensor(mb_end-mb_start+1, 2*data.contextSize)
  local m = 1
  local n = 1 
  for mk, mv in pairs(contextid_minibatch) do
      n = 1
      for nk, nv in pairs(contextid_minibatch[mk]) do
           cid[{{m}, {n}}] = contextid_minibatch[mk][nk]
           lab[{{m},{n}}] = labels[mk][nk]
           n = n + 1
      end
      wid[{{m},{1}}] = wordid_minibatch[mk]
      m = m + 1
  end
  inputs = {cid, wid}
  --print(cid)
  --print(wid)
  --print(contextid_minibatch) 
  -- inputs = {contextid_minibatch,wordid_minibatch}
  --inputs = torch.Tensor(inputs)
  --labels = torch.Tensor(labels)
  --print(inputs)
  -- print(lab)
  -- return inputs, labels
  return cid,wid, lab
  --return {contextid_minibatch, wordid_minibatch}
end

return context
