--
-- Created by IntelliJ IDEA.
-- User: asebag
-- Date: 7/30/18
-- Time: 12:46 PM
-- To change this template use File | Settings | File Templates.
--

local OuterProduct, parent = torch.class('nn.OuterProduct', 'nn.Module')

function OuterProduct:__init()
   parent.__init(self)
   self.gradInput = {torch.Tensor(), torch.Tensor() }
end 
 
function OuterProduct:updateOutput(input)
   local input1, input2 = input[1], input[2]
   if input1:dim() == 1 then
      -- convert non batch input to batch input
      input1 = input1:view(1,-1)
      input2 = input2:view(1,-1)
   end
   local nbatch = input1:size(1)
   if self.output:nElement()==0 then
      self.output = torch.Tensor(nbatch, input1:size(2), input2:size(2)):typeAs(input1)
   end
   for k=1, nbatch do
       self.output[k]:ger(input1[k], input2[k])
   end
   return self.output
end

function OuterProduct:updateGradInput(input, gradOutput)
   local v1 = input[1]
   local v2 = input[2]
   local not_batch = false
   
   if #self.gradInput ~= 2 or self.gradInput[1]:nElement()==0 or self.gradInput[2]:nElement()==0 then
     self.gradInput[1] = input[1].new(input[1]:size())
     self.gradInput[2] = input[2].new(input[2]:size())
   end
   self.gradInput[1]:zero()
   self.gradInput[2]:zero()

   if v1:dim() == 1 then
      v1 = v1:view(1,-1)
      v2 = v2:view(1,-1)
      not_batch = true
   end
   
   --Taking care of grad wrt v2
   local gw2 = self.gradInput[2]
   gw2:resize(v2:size(1), 1, v2:size(2))
   v1 = v1:resize(v1:size(1), 1, v1:size(2))
   for k=1, v2:size(1) do
      gw2[k]:addmm(v1[k], gradOutput[k])
   end
   --Putting stuff back into correct shape
   gw2:resizeAs(v2)
   v1:resize(v1:size(1), v1:size(3))

   --Taking care of grad wrt v1
   v2 = v2:resize(v2:size(1), v2:size(2), 1)
   local gw1 = self.gradInput[1]
   gw1:resize(v1:size(1), v1:size(2), 1)
   for k=1, v1:size(1) do
      gw1[k]:addmm(gradOutput[k], v2[k])
   end
   --Putting stuff back into correct shape
   gw1:resizeAs(v1)
   v2:resize(v2:size(1), v2:size(2))

   if not_batch then
      -- unbatch gradInput
      self.gradInput[1]:set(gw1:select(1,1))
      self.gradInput[2]:set(gw2:select(1,1))
   end

   return self.gradInput
end

function OuterProduct:clearState()
   return parent.clearState(self)
end


