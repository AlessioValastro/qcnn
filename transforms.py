import torch

def convert_to_quaternion(image):
    """
    Converte un'immagine RGB (3, H, W) in quaternione
    con forma (4, H, W).
    """
    # Separo i canali RGB
    r, g, b = image[0], image[1], image[2]

    # Calcolo la luminosità come quarta componente con formula ponderata
    lum = 0.299 * r + 0.587 * g + 0.114 * b

    # Stack per ottenere (4, H, W)
    quaternion_image = torch.stack([lum, r, g, b], dim=0)

    return quaternion_image  # (4, H, W)

class ToQuaternion:
    """
    Classe per poter chiamare la funzione convert_to_quaternion
    come una trasformazione di PyTorch.
    """
    def __call__(self, image):
        return convert_to_quaternion(image)

    def __repr__(self):
        return self.__class__.__name__ + '()'