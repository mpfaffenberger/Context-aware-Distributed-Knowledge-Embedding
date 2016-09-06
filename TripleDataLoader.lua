local TripleDataLoader = torch.class('TripleDataLoader')

function TripleDataLoader:__init(datafile, batchSize, negSize, logger)
    -- class variables
    local data = torch.load(datafile)
    self.subs = data.subs
    self.rels = data.rels
    self.objs = data.objs
    self.numPredicates = data.numPredicates
    self.numEntities = data.numEntities

    -- additional variables
    self.batchSize = batchSize
    self.negSize = negSize
    self.numData = self.subs:size(1)

    self.entRange = math.max(torch.max(self.subs), torch.max(self.objs))
    self.relRange = torch.max(self.rels)

    -- allocate memory
    self.repsubs = torch.LongTensor(self.batchSize * self.negSize)
    self.reprels = torch.LongTensor(self.batchSize * self.negSize)
    self.repobjs = torch.LongTensor(self.batchSize * self.negSize)

    self.negsubs = torch.LongTensor(self.batchSize * self.negSize)
    self.negrels = torch.LongTensor(self.batchSize * self.negSize)
    self.negobjs = torch.LongTensor(self.batchSize * self.negSize)

    if logger then
        self.logger = logger
        self.logger.info(string.rep('-', 50))
        self.logger.info(string.format('TripleDataLoader Configurations:'))
        self.logger.info(string.format('    number of data : %d', self.numData))
        self.logger.info(string.format('    mini batch size: %d', self.batchSize))
        self.logger.info(string.format('    entity id range: %d', self.entRange))
        self.logger.info(string.format('    relatn id range: %d', self.relRange))
    end
end

function TripleDataLoader:nextBatch(circular)
    if self.currIdx == nil or self.currIdx + self.batchSize > self.numData then
        self.currIdx = 1
        self.indices = torch.LongTensor.torch.randperm(self.numData)
    end

    -- select data indices
    local dataIdx = self.indices:narrow(1, self.currIdx, self.batchSize)
    self.currIdx = self.currIdx + self.batchSize

    -- positive samples
    self.repsubs = torch.repeatTensor(self.repsubs, self.subs:index(1, dataIdx), self.negSize)
    self.reprels = torch.repeatTensor(self.reprels, self.rels:index(1, dataIdx), self.negSize)
    self.repobjs = torch.repeatTensor(self.repobjs, self.objs:index(1, dataIdx), self.negSize)

    -- negative samples
    self.negsubs:random(1, self.entRange)
    self.negsubs:maskedSelect(torch.eq(self.negsubs, self.repsubs)):random(1, self.entRange)

    self.negobjs:random(1, self.entRange)
    self.negobjs:maskedSelect(torch.eq(self.negobjs, self.repobjs)):random(1, self.entRange)

    self.negrels:random(1, self.relRange)
    self.negrels:maskedSelect(torch.eq(self.negrels, self.reprels)):random(1, self.relRange)

    return cudacheck(self.repsubs), cudacheck(self.reprels), cudacheck(self.repobjs), cudacheck(self.negsubs), cudacheck(self.negobjs), cudacheck(self.negrels)
end

-- create torch-format data for TripleDataLoader
function createTripleData(dataPath, savePath)

    local counter = 1
    local tokens = {}

    -- class variables
    local subs = {}
    local objs = {}
    local rels = {}
    local dict = {}
    local preds = {}

    -- read data fileh
    local file = io.open(dataPath, 'r')
    local line

    function maybeAdd(str)
        if tokens[str] == nil then
            tokens[str] = counter
            counter = counter + 1;
        end
    end

    while true do
        line = file:read()
        if line == nil then break end
        local triples = stringx.split(line)
        maybeAdd(triples[1])
        maybeAdd(triples[2])
        maybeAdd(triples[3])
    end

    local file = io.open(dataPath, 'r')
    local line
    
    -- for each line of triple data
    while true do
        line = file:read()
        if line == nil then break end
        local fields = stringx.split(line)

        preds[tokens[fields[2]]] = "_"
        subs[#subs+1] = tokens[fields[1]]
        rels[#rels+1] = tokens[fields[2]]
        objs[#objs+1] = tokens[fields[3]]
        dict[table.concat({subs[#subs], rels[#rels], objs[#objs]}, '_')] = true
        
    end
    file:close()

    local data = {}
    data.subs = torch.LongTensor(subs)
    data.rels = torch.LongTensor(rels)
    data.objs = torch.LongTensor(objs)
    data.dict = dict
    data.numEntities = counter - 1
    data.numPredicates = #rels
    print(#preds)
    print(counter - 1)
    torch.save(savePath, data)
end