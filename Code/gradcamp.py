def gradcamp(model, dataset, device, class_names, image_size=(128, 128)):
    import matplotlib.pyplot as plt
    import torchvision.transforms as transforms
    import torch.nn.functional as F
    import os
    import cv2
    import numpy as np
    from PIL import Image

    model.eval()
    transform = transforms.Compose([
        transforms.Resize(image_size),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225])
    ])

    num_classes_to_show = min(5, len(class_names))
    fig, axes = plt.subplots(num_classes_to_show, 3, figsize=(15, 3 * num_classes_to_show))
    
    if num_classes_to_show == 1:
        axes = np.expand_dims(axes, axis=0)  # Ensure axes[i][j] indexing works

    for row, class_name in enumerate(class_names[:num_classes_to_show]):
        class_dir = os.path.join(dataset.root_dir, class_name)
        sample_img_name = os.listdir(class_dir)[0]
        img_path = os.path.join(class_dir, sample_img_name)
        
        image = Image.open(img_path).convert('RGB')
        image_tensor = transform(image).unsqueeze(0).to(device)

        # Hook target layer
        target_layer = model.conv_layers[-2]
        activations = None
        gradients = None

        def forward_hook(module, input, output):
            nonlocal activations
            activations = output

        def backward_hook(module, grad_input, grad_output):
            nonlocal gradients
            gradients = grad_output[0]

        h1 = target_layer.register_forward_hook(forward_hook)
        h2 = target_layer.register_backward_hook(backward_hook)

        # Forward + backward pass
        output = model(image_tensor)
        pred = output.argmax(dim=1).item()

        model.zero_grad()
        output[0, pred].backward()

        pooled_gradients = torch.mean(gradients, dim=[0, 2, 3])
        for i in range(activations.shape[1]):
            activations[:, i, :, :] *= pooled_gradients[i]
        
        heatmap = torch.mean(activations, dim=1).squeeze()
        heatmap = torch.relu(heatmap)
        heatmap /= torch.max(heatmap) + 1e-8
        heatmap = heatmap.detach().cpu().numpy()
        heatmap = cv2.resize(heatmap, image_size)

        # Convert images
        orig_img = np.array(image.resize(image_size))
        heatmap_img = np.uint8(255 * heatmap)
        heatmap_color = cv2.applyColorMap(heatmap_img, cv2.COLORMAP_JET)
        superimposed_img = cv2.addWeighted(orig_img, 0.6, heatmap_color, 0.4, 0)

        # Show images
        axes[row][0].imshow(orig_img)
        axes[row][0].axis('off')
        axes[row][0].set_title(f"Original\n{class_name}")

        axes[row][1].imshow(heatmap_color)
        axes[row][1].axis('off')
        axes[row][1].set_title("Grad-CAM")

        axes[row][2].imshow(superimposed_img)
        axes[row][2].axis('off')
        axes[row][2].set_title(f"Predicted: {class_names[pred]}")

        h1.remove()
        h2.remove()

    plt.tight_layout()
    plt.suptitle("Grad-CAM Visualization", fontsize=18, y=1.03)
    plt.savefig("gcPlantVillage.png", dpi=300, bbox_inches='tight')
    plt.show()
