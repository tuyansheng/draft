import os
from torchvision.datasets import MNIST, CIFAR10, CIFAR100
from torchvision.transforms import ToPILImage

def export_dataset(dataset, out_root):
    to_pil = ToPILImage()
    os.makedirs(out_root, exist_ok=True)

    for idx, (img, label) in enumerate(dataset):
        class_dir = os.path.join(out_root, str(label))
        os.makedirs(class_dir, exist_ok=True)

        img = to_pil(img)
        img.save(os.path.join(class_dir, f"{idx}.png"))

def export_all(image_root="~/ImageData"):
    image_root = os.path.expanduser(image_root)

    # ========= MNIST =========
    print("Exporting MNIST...")
    mnist_train = MNIST(root=image_root, train=True, download=True)
    mnist_test  = MNIST(root=image_root, train=False, download=True)

    export_dataset(mnist_train, os.path.join(image_root, "mnist", "Train"))
    export_dataset(mnist_test,  os.path.join(image_root, "mnist", "Test"))

    # ========= CIFAR-10 =========
    print("Exporting CIFAR-10...")
    cifar10_train = CIFAR10(root=image_root, train=True, download=True)
    cifar10_test  = CIFAR10(root=image_root, train=False, download=True)

    export_dataset(cifar10_train, os.path.join(image_root, "cifar10", "Train"))
    export_dataset(cifar10_test,  os.path.join(image_root, "cifar10", "Test"))

    # ========= CIFAR-100 =========
    print("Exporting CIFAR-100...")
    cifar100_train = CIFAR100(root=image_root, train=True, download=True)
    cifar100_test  = CIFAR100(root=image_root, train=False, download=True)

    export_dataset(cifar100_train, os.path.join(image_root, "cifar100", "Train"))
    export_dataset(cifar100_test,  os.path.join(image_root, "cifar100", "Test"))

    print("All datasets exported successfully.")

if __name__ == "__main__":
    export_all()
