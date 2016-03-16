local ImageNetClasses = torch.load('./ImageNetClasses')
for i=1001,#ImageNetClasses.ClassName do
    ImageNetClasses.ClassName[i] = nil
end

function Key(num)
    return string.format('%07d',num)
end


return
{
    TRAINING_PATH = '/var/scratch/agrotov/ilsvrc2012train/ImageData/', --Training images location
    VALIDATION_PATH = '/var/scratch/agrotov/ilsvrc2012vali/ImageData/',  --Validation images location
    VALIDATION_DIR = '/var/scratch/agrotov/ilsvrc2012vali/ImageData/', --Validation LMDB location
    TRAINING_DIR = '/var/scratch/agrotov/ilsvrc2012train/ImageData/', --Training LMDB location
    ImageMinSide = 256, --Minimum side length of saved images
    ValidationLabels = torch.load('/var/scratch/agrotov/ilsvrc2012vali/ILSVRC2014_clsloc_validation_ground_truth.txt'),
    ImageNetClasses = ImageNetClasses,
    Normalization = {'simple', 118.380948, 61.896913}, --Default normalization -global mean, std
    Compressed = true,
    Key = Key
}
