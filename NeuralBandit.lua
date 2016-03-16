--
-- Created by IntelliJ IDEA.
-- User: agrotov
-- Date: 3/11/16
-- Time: 12:35 PM
-- To change this template use File | Settings | File Templates.
--

require 'distributions'
local DataProvider = require 'DataProvider'

local GeneralizedSequential, _ = torch.class('nn.GeneralizedSequential', 'nn.Sequential')

function GeneralizedSequential:forward(input, target)
    return self:updateOutput(input, target)
end

function GeneralizedSequential:updateOutput(input, target)
    local currentOutput = input
    for i=1,#self.modules do
        currentOutput = self.modules[i]:updateOutput(currentOutput, target)
    end
    self.output = currentOutput
    return currentOutput
end



local function sample_action(model_output)
    local result = torch.multinomial(model_output,1)
    return result
end


local function reward_for_actions(actions, labels)

end

local function probability_of_actions(model_output, actions)
    return model_output:gather(2,actions)
end

local function save_data_set(actions, rewards, probabilities)
end

local function bandit_loss(model_output, actions, rewards, probabilities,loss)
    loss:forward(y,yt)
    df_dx = model_output
    return -model_output.gather(2,actions)*rewards/probabilities
end

function MyCrossEntropyCriterion(weights)
    criterion = nn.GeneralizedSequential()
    criterion:add(nn.LogSoftMax())
    criterion:add(nn.ClassNLLCriterion(weights))
    return criterion
end

function BanditCriterion(weights)
    criterion = nn.GeneralizedSequential()
    criterion:add(nn.SoftMax())
    criterion:add(nn.ClassNLLCriterion(weights))
    return criterion
end


local function train_with_bandits(DB, config, model_teacher, model_student)


    loss = nn.Sequential()
    loss.add(nn.Mul())
    local optimizer = Optimizer{
    Model = model,
    Loss = bandit_loss,
    OptFunction = _G.optim[opt.optimization],
    OptState = optimState,
    Parameters = {Weights, Gradients},
    Regime = trainRegime
    }

    confusion:zero()

    local SizeData = DB:size()
    if not AllowVarBatch then SizeData = math.floor(SizeData/opt.batchSize)*opt.batchSize end
    local dataIndices = torch.range(1, SizeData, opt.bufferSize):long()
    if train and opt.shuffle then --shuffle batches from LMDB
        dataIndices = dataIndices:index(1, torch.randperm(dataIndices:size(1)):long())
    end

    local numBuffers = 2
    local currBuffer = 1
    local BufferSources = {}
    for i=1,numBuffers do
        BufferSources[i] = DataProvider.Container{
            Source = {torch.ByteTensor(),torch.IntTensor()}
        }
    end


    local currBatch = 1

    local BufferNext = function()
        currBuffer = currBuffer%numBuffers +1
        if currBatch > dataIndices:size(1) then BufferSources[currBuffer] = nil return end
        local sizeBuffer = math.min(opt.bufferSize, SizeData - dataIndices[currBatch]+1)
        BufferSources[currBuffer].Data:resize(sizeBuffer ,unpack(config.InputSize))
        BufferSources[currBuffer].Labels:resize(sizeBuffer)
        DB:asyncCacheSeq(config.Key(dataIndices[currBatch]), sizeBuffer, BufferSources[currBuffer].Data, BufferSources[currBuffer].Labels)
        currBatch = currBatch + 1
    end

    local MiniBatch = DataProvider.Container{
        Name = 'GPU_Batch',
        MaxNumItems = opt.batchSize,
        Source = BufferSources[currBuffer],
        TensorType = TensorType
    }


    local yt = MiniBatch.Labels
    local y = torch.Tensor()
    local x = MiniBatch.Data
    local NumSamples = 0
    local lossVal = 0
    local currLoss = 0

    BufferNext()

    while NumSamples < SizeData do
        DB:synchronize()
        MiniBatch:reset()
        MiniBatch.Source = BufferSources[currBuffer]
        if train and opt.shuffle then MiniBatch.Source:shuffleItems() end
        BufferNext()

        while MiniBatch:getNextBatch() do
            if #normalization>0 then MiniBatch:normalize(unpack(normalization)) end
            if train then
                y = model_teacher:forward(x)
                actions = sample_action(y)
                probabilities = probability_of_actions(actions)
                rewards = reward_for_actions(actions)

                y, currLoss = optimizer:optimize(x, yt)

                if opt.nGPU > 1 then
                    model:syncParameters()
                end
            else
                y = model:forward(x)
                currLoss = loss:forward(y,yt)
            end
            lossVal = currLoss + lossVal
            if type(y) == 'table' then --table results - always take first prediction
                y = y[1]
            end
            confusion:batchAdd(y,yt)
            NumSamples = NumSamples + x:size(1)
            xlua.progress(NumSamples, SizeData)
        end

        if train and opt.checkpoint >0 and (currBatch % math.ceil(opt.checkpoint/opt.bufferSize) == 0) then
            print(NumSamples)
            confusion:updateValids()
            print('\nAfter ' .. NumSamples .. ' samples, current error is: ' .. 1-confusion.totalValid .. '\n')
            torch.save(netFilename .. '_checkpoint' .. '.t7', savedModel)
        end
        collectgarbage()
    end
    xlua.progress(NumSamples, SizeData)
    return(lossVal/math.ceil(SizeData/opt.batchSize))
end

local function produce_bandit_data(DB, train, config)

    local SizeData = DB:size()
    if not AllowVarBatch then SizeData = math.floor(SizeData/opt.batchSize)*opt.batchSize end
    local dataIndices = torch.range(1, SizeData, opt.bufferSize):long()

    local numBuffers = 2
    local currBuffer = 1
    local BufferSources = {}
    for i=1,numBuffers do
        BufferSources[i] = DataProvider.Container{
            Source = {torch.ByteTensor(),torch.IntTensor()}
        }
    end


    local currBatch = 1

    local BufferNext = function()
        currBuffer = currBuffer%numBuffers +1
        if currBatch > dataIndices:size(1) then BufferSources[currBuffer] = nil return end
        local sizeBuffer = math.min(opt.bufferSize, SizeData - dataIndices[currBatch]+1)
        BufferSources[currBuffer].Data:resize(sizeBuffer ,unpack(config.InputSize))
        BufferSources[currBuffer].Labels:resize(sizeBuffer)
        DB:asyncCacheSeq(config.Key(dataIndices[currBatch]), sizeBuffer, BufferSources[currBuffer].Data, BufferSources[currBuffer].Labels)
        currBatch = currBatch + 1
    end

    local MiniBatch = DataProvider.Container{
        Name = 'GPU_Batch',
        MaxNumItems = opt.batchSize,
        Source = BufferSources[currBuffer],
        TensorType = TensorType
    }


    local yt = MiniBatch.Labels
    local y = torch.Tensor()
    local x = MiniBatch.Data
    local NumSamples = 0

    BufferNext()

    while NumSamples < SizeData do
        DB:synchronize()
        MiniBatch:reset()
        MiniBatch.Source = BufferSources[currBuffer]
        BufferNext()

        while MiniBatch:getNextBatch() do
            if #normalization>0 then MiniBatch:normalize(unpack(normalization)) end
            y = model:forward(x)
            actions = sample_action(y)
            probabilities = probability_of_actions(actions)
            rewards = reward_for_actions(actions)



            if type(y) == 'table' then --table results - always take first prediction
                y = y[1]
            end
            confusion:batchAdd(y,yt)
            NumSamples = NumSamples + x:size(1)
            xlua.progress(NumSamples, SizeData)
        end


        collectgarbage()
    end
    xlua.progress(NumSamples, SizeData)
    return 0
end

