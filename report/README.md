# Report
This is work on a report.

Algorithm: Training DiT Model Using DeepSpeed

Input: Command-line arguments (args)
Output: Trained DiT model

1. Initialize Distributed Environment:
   - Initialize DeepSpeed distributed training.
   - Set rank, device, and seed based on global seed and world size.
   - Set CUDA device.

2. Create Experiment Directory:
   - Create an experiment directory and checkpoint directory if rank is 0.
   - Create a logger to log training progress.

3. Create Model:
   - Ensure image size is divisible by 8.
   - Initialize the DiT model with specified parameters.
   - Load pretrained weights if resuming from a checkpoint.

4. Initialize Diffusion and VAE:
   - Create diffusion model with default settings.
   - Load VAE model and move it to the appropriate device.

5. Prepare Data:
   - Define image transformation pipeline.
   - Load dataset and create a distributed sampler.
   - Create data loader with specified batch size and other parameters.

6. Initialize DeepSpeed Model Engine:
   - Initialize DeepSpeed model engine with the model and optimizer.

7. Training Loop:
   - For each epoch:
     a. Set epoch for the distributed sampler.
     b. For each batch of data (x, y):
        i. Move data to the appropriate device.
        ii. Encode images to latent space using VAE.
        iii. Sample timesteps for diffusion.
        iv. Compute training losses using the diffusion model.
        v. Backpropagate the loss and update model parameters.
        vi. Log training loss and speed periodically.
        vii. Save model checkpoint periodically.

8. Cleanup:
   - Destroy the distributed process group.

Function Definitions:

Function requires_grad(model, flag):
   - Set requires_grad flag for all parameters in the model.

Function cleanup():
   - Destroy the distributed process group.

Function create_logger(logging_dir):
   - Create a logger that writes to a log file and stdout if rank is 0, otherwise create a dummy logger.

Function center_crop_arr(pil_image, image_size):
   - Center crop the image to the specified size.

Main Function:

Function main(args):
   - Perform steps 1 to 8 as described above.

If __name__ == "__main__":
   - Parse command-line arguments.
   - Call main function with parsed arguments.
