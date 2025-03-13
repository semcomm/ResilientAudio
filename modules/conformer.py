import torch
from .conformer_modules import ConvolutionModule
from .conformer_layers import EncoderLayer, MultiHeadedAttention
from .conformer_modules import LayerNorm
from .conformer_modules import MultiLayeredConv1d
from .conformer_modules import RotaryEmbedding
from .conformer_modules import Swish
from .conformer_modules import repeat



class Conformer(torch.nn.Module):
    """
    Conformer encoder module.
    Args:
        idim (int): Input dimension.
        attention_dim (int): Dimension of attention.
        attention_heads (int): The number of heads of multi head attention.
        linear_units (int): The number of units of position-wise feed forward.
        num_blocks (int): The number of decoder blocks.
        dropout_rate (float): Dropout rate.
        positional_dropout_rate (float): Dropout rate after adding positional encoding.
        attention_dropout_rate (float): Dropout rate in attention.
        input_layer (Union[str, torch.nn.Module]): Input layer type.
        normalize_before (bool): Whether to use layer_norm before the first block.
        concat_after (bool): Whether to concat attention layer's input and output.
            if True, additional linear will be applied.
            i.e. x -> x + linear(concat(x, att(x)))
            if False, no additional linear will be applied. i.e. x -> x + att(x)
        positionwise_layer_type (str): "linear", "conv1d", or "conv1d-linear".
        positionwise_conv_kernel_size (int): Kernel size of positionwise conv1d layer.
        macaron_style (bool): Whether to use macaron style for positionwise layer.
        pos_enc_layer_type (str): Conformer positional encoding layer type.
        selfattention_layer_type (str): Conformer attention layer type.
        activation_type (str): Conformer activation function type.
        use_cnn_module (bool): Whether to use convolution module.
        cnn_module_kernel (int): Kernerl size of convolution module.
        padding_idx (int): Padding idx for input_layer=embed.
    """

    def __init__(self, attention_dim=256, attention_heads=4, linear_units=2048, num_blocks=6, dropout_rate=0.1, attention_dropout_rate=0.0,
                 normalize_before=True, concat_after=False, positionwise_conv_kernel_size=1, macaron_style=False,
                 use_cnn_module=False, cnn_module_kernel=31):
        super(Conformer, self).__init__()

        activation = Swish()
        self.conv_subsampling_factor = 1

        self.rotary_emb = RotaryEmbedding(attention_dim//attention_heads)

        self.normalize_before = normalize_before


        # self-attention module definition
        encoder_selfattn_layer = MultiHeadedAttention
        encoder_selfattn_layer_args = (attention_heads, attention_dim, attention_dropout_rate)

        # feed-forward module definition
        positionwise_layer = MultiLayeredConv1d
        positionwise_layer_args = (attention_dim, linear_units, positionwise_conv_kernel_size, dropout_rate,)

        # convolution module definition
        convolution_layer = ConvolutionModule
        convolution_layer_args = (attention_dim, cnn_module_kernel, activation)

        self.encoders = repeat(num_blocks, lambda lnum: EncoderLayer(attention_dim, encoder_selfattn_layer(*encoder_selfattn_layer_args),
                                                                     positionwise_layer(*positionwise_layer_args),
                                                                     positionwise_layer(*positionwise_layer_args) if macaron_style else None,
                                                                     convolution_layer(*convolution_layer_args) if use_cnn_module else None, dropout_rate,
                                                                     normalize_before, concat_after))
        if self.normalize_before:
            self.after_norm = LayerNorm(attention_dim)


    def forward(self, xs, masks, embeds = None):
        """
        Encode input sequence.
        Args:
            utterance_embedding: embedding containing lots of conditioning signals
            step: indicator for when to start updating the embedding function
            xs (torch.Tensor): Input embeddings tensor (#batch, time, idim).
            masks (torch.Tensor): Mask tensor (#batch, time).
        Returns:
            torch.Tensor: Output tensor (#batch, time, attention_dim).
            torch.Tensor: Mask tensor (#batch, time).
        """

        rotary_emb = self.rotary_emb(xs.shape[-2])

        # cache = xs[:,:-1,:]
        xs, masks, _, _, _ = self.encoders(xs, masks, None, embeds, rotary_emb)

        # xs, masks, _, _, _ = self.encoders(xs, masks, cache, embeds, rotary_emb)
        # if isinstance(xs, tuple):
        #     xs = xs[0]


        if self.normalize_before:
            xs = self.after_norm(xs)

        return xs, masks


class StreamingTransformer(torch.nn.Module):
    """
    Conformer encoder module.
    Args:
        idim (int): Input dimension.
        attention_dim (int): Dimension of attention.
        attention_heads (int): The number of heads of multi head attention.
        linear_units (int): The number of units of position-wise feed forward.
        num_blocks (int): The number of decoder blocks.
        dropout_rate (float): Dropout rate.
        positional_dropout_rate (float): Dropout rate after adding positional encoding.
        attention_dropout_rate (float): Dropout rate in attention.
        input_layer (Union[str, torch.nn.Module]): Input layer type.
        normalize_before (bool): Whether to use layer_norm before the first block.
        concat_after (bool): Whether to concat attention layer's input and output.
            if True, additional linear will be applied.
            i.e. x -> x + linear(concat(x, att(x)))
            if False, no additional linear will be applied. i.e. x -> x + att(x)
        positionwise_layer_type (str): "linear", "conv1d", or "conv1d-linear".
        positionwise_conv_kernel_size (int): Kernel size of positionwise conv1d layer.
        macaron_style (bool): Whether to use macaron style for positionwise layer.
        pos_enc_layer_type (str): Conformer positional encoding layer type.
        selfattention_layer_type (str): Conformer attention layer type.
        activation_type (str): Conformer activation function type.
        use_cnn_module (bool): Whether to use convolution module.
        cnn_module_kernel (int): Kernerl size of convolution module.
        padding_idx (int): Padding idx for input_layer=embed.
    """

    def __init__(self, attention_dim=256, attention_heads=4, linear_units=2048, num_blocks=6, dropout_rate=0.1, attention_dropout_rate=0.0,
                 normalize_before=True, concat_after=False, positionwise_conv_kernel_size=1, macaron_style=False,
                 use_cnn_module=False, cnn_module_kernel=31):
        super(StreamingTransformer, self).__init__()

        activation = Swish()
        self.conv_subsampling_factor = 1

        self.rotary_emb = RotaryEmbedding(attention_dim//attention_heads)

        self.normalize_before = normalize_before


        # self-attention module definition
        encoder_selfattn_layer = MultiHeadedAttention
        encoder_selfattn_layer_args = (attention_heads, attention_dim, attention_dropout_rate)

        # feed-forward module definition
        positionwise_layer = MultiLayeredConv1d
        positionwise_layer_args = (attention_dim, linear_units, positionwise_conv_kernel_size, dropout_rate,)

        # convolution module definition
        convolution_layer = ConvolutionModule
        convolution_layer_args = (attention_dim, cnn_module_kernel, activation)

        self.encoders = repeat(num_blocks, lambda lnum: EncoderLayer(attention_dim, encoder_selfattn_layer(*encoder_selfattn_layer_args),
                                                                     positionwise_layer(*positionwise_layer_args),
                                                                     positionwise_layer(*positionwise_layer_args) if macaron_style else None,
                                                                     convolution_layer(*convolution_layer_args) if use_cnn_module else None, dropout_rate,
                                                                     normalize_before, concat_after))
        if self.normalize_before:
            self.after_norm = LayerNorm(attention_dim)


    def forward(self, xs, masks, embeds = None):
        """
        Encode input sequence.
        Args:
            utterance_embedding: embedding containing lots of conditioning signals
            step: indicator for when to start updating the embedding function
            xs (torch.Tensor): Input embeddings tensor (#batch, time, idim).
            masks (torch.Tensor): Mask tensor (#batch, time).
        Returns:
            torch.Tensor: Output tensor (#batch, time, attention_dim).
            torch.Tensor: Mask tensor (#batch, time).
        """

        rotary_emb = self.rotary_emb(xs.shape[-2])

        xs, masks, _, _, _ = self.encoders(xs, masks, None, embeds, rotary_emb)

        # cache = xs[:,:-1,:]
        # xs, masks, _, _, _ = self.encoders(xs, masks, cache, embeds, rotary_emb)

        if self.normalize_before:
            xs = self.after_norm(xs)

        return xs, masks
