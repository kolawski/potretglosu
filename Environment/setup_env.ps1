$dockerfilePath = ".\Environment\Dockerfile"

$imageName = "voice-reproduction"
$buildCommand = "docker build --progress=plain -t $imageName -f $dockerfilePath ."
& cmd /c $buildCommand

Write-Host "Image was built: $imageName"
