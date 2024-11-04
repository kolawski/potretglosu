$imageName = "voice-reproduction"

try {

    $hostDirectory = Get-Location
    $containerDirectory = "/app"

    $containerName = "voice-reproduction-container"
    $runCommand = "docker run --rm -it --gpus all -p 8050:8050 --entrypoint /bin/bash -v `"$hostDirectory`:$containerDirectory`" -w `"$containerDirectory`" --name $containerName $imageName"
    & cmd /c $runCommand

} catch {
    Write-Host "Error: $_"
    Write-Host "Image '$imageName' may be unavailable. Please, run setup_env.ps1 script."
}
