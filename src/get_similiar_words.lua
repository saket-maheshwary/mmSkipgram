
get_similiar_words = {}

-- function get_similiar_words.print_sim_words(mod,words, k)
--     for i = 1, #words do
--     	r = get_sim_words(mod,words[i], k)
-- 	if r ~= nil then
--    	    print("-------"..words[i].."-------")
-- 	    for j = 1, k do
-- 	        print(string.format("%s, %.4f", r[j][1], r[j][2]))
-- 	    end
-- 	end
--     end
-- end
-- 
local function normalize(m)
    m_norm = torch.zeros(m:size())
    for i = 1, m:size(1) do
    	m_norm[i] = m[i] / torch.norm(m[i])
    end
    return m_norm
end

function  get_similiar_words.get_sim_words(mod,w, k)
   
    local word_vecs_norm = normalize(mod.word_vecs.weight:double())
    
    if type(w) == "string" then
        if mod.word2id[w] == nil then
	   print("'"..w.."' does not exist in vocabulary.")
	   return nil
	else
            w2 = word_vecs_norm[mod.word2id[w]]
	end
    end
    local sim = torch.mv(word_vecs_norm, w2)
    sim, idx = torch.sort(-sim)
    local r = {}
    for i = 1, k do
        r[i] = {mod.id2word[idx[i]], -sim[i]}
    end
    
    if r ~= nil then
   	    print(string.format("-------%s-------",w))
       for j = 1, k do
         print(string.format("%s, %.4f", r[j][1], r[j][2]))
       end
     end
   return r
end

return get_similiar_words
