import tensorflow as tf
import tensorflow.keras.backend as K
import tensorflow.keras.layers as keras_layers

from ... import layers as ilayers

#---------------------------------------------------FORWARD PASS--------------------------------------------------------

#------------BASICS
@tf.function
def out_func(ins, layer_func):
    #print("Tracing out_func...", ins, layer_func)
    return layer_func(ins)

@tf.function
def final_out_func(ins, layer_func, final_mapping, neuron_selection, r_init):
    #print("Tracing final out_func...", tf.shape(ins), neuron_selection, r_init)
    output = layer_func(ins)
    output = final_mapping(output, neuron_selection, r_init)
    return output

@tf.function
def neuron_select(Ys, neuron_selection):
    """
    Performs neuron_selection on Ys

    :param Ys: output of own forward pass
    :param neuron_selection: neuron_selection parameter (see try_apply)
    """
    #error handling is done before, in try_apply
    #print("Tracing neuron_select", "Ys: ", type(Ys), Ys.shape, " neuron_selection: ", type(neuron_selection), neuron_selection)

    if neuron_selection is None:
        Ys = Ys
    elif isinstance(neuron_selection, tf.Tensor):
        # flatten and then filter neuron_selection index
        Ys = tf.reshape(Ys, (Ys.shape[0], -1))
        Ys = tf.gather_nd(Ys, neuron_selection, batch_dims=1)
    else:
        Ys = K.max(Ys, axis=1, keepdims=True)
    return Ys

#------------GRADIENT BASED

#------------LRP

#---------------------------------------------------EXPLANATIONS--------------------------------------------------------

#------------BASICS
@tf.function
def base_explanation(reversed_outs, n_ins, n_outs):
    """
    function that computes the explanations.
    * Core XAI functionality

    :param reversed_outs: either backpropagated explanation(s) of child layers, or None if this is the last layer
    :param n_ins: int, (expected) number of inputs
    :param n_outs: int, (expected) number of outputs

    :returns explanation, or tensor of multiple explanations if the layer has multiple inputs (one for each)
    """
    #print("Tracing base explanation")
    # TODO Leander: consider all cases
    if n_outs > 1:
        ret = keras_layers.Add(dtype=tf.float32)([r for r in reversed_outs])
    elif n_ins > 1:
        ret = [reversed_outs for i in range(n_ins)]
        ret = tf.keras.layers.concatenate(ret, axis=1)
    else:
        ret = reversed_outs
    return ret

#------------GRADIENT BASED

@tf.function
def gradient_explanation(ins, layer_func, out_func, reversed_outs, n_ins, n_outs):
    #print("Tracing gradient explanation", ins, layer_func, reversed_outs)

    outs = out_func(ins, layer_func)

    # correct number of outs
    if n_outs > 1:
        outs = [outs for _ in range(n_outs)]

    if n_outs > 1:
        if n_ins > 1:
            ret = [keras_layers.Add(dtype=tf.float32)(
                [tf.gradients(o, i, grad_ys=r)[0] for o, r in zip(outs, reversed_outs)]) for i in ins]
        else:
            ret = keras_layers.Add(dtype=tf.float32)(
                [tf.gradients(o, ins, grad_ys=r)[0] for o, r in zip(outs, reversed_outs)])
    else:
        if n_ins > 1:
            ret = [tf.gradients(outs, i, grad_ys=reversed_outs)[0] for i in ins]
        else:
            ret = tf.gradients(outs, ins, grad_ys=reversed_outs)[0]

    return ret

@tf.function
def final_gradient_explanation(ins, layer_func, final_mapping, out_func, reversed_outs, n_ins, n_outs, neuron_selection, r_init):
    #print("Tracing final gradient explanation", ins)

    outs = tf.squeeze(out_func(ins, layer_func, final_mapping, neuron_selection, r_init))

    # correct number of outs
    if n_outs > 1:
        outs = [outs for _ in range(n_outs)]

    if n_outs > 1:
        if n_ins > 1:
            ret = [keras_layers.Add(dtype=tf.float32)(
                [tf.gradients(o, i, grad_ys=r)[0] for o, r in zip(outs, reversed_outs)]) for i in ins]
        else:
            ret = keras_layers.Add(dtype=tf.float32)(
                [tf.gradients(o, ins, grad_ys=r)[0] for o, r in zip(outs, reversed_outs)])
    else:
        if n_ins > 1:
            ret = [tf.gradients(outs, i, grad_ys=reversed_outs)[0] for i in ins]
        else:
            ret = tf.gradients(outs, ins, grad_ys=reversed_outs)[0]

    return ret

@tf.function
def deconvnet_explanation(ins, layer_func, layer_func_wo_relu, activation_func, out_func, reversed_outs, n_ins, n_outs):

    #print("TRACING DECONV NET")

    outs = out_func(ins, layer_func)

    # Apply relus conditioned on backpropagated values.
    Ys_wo_relu = out_func(ins, layer_func_wo_relu)

    if n_outs > 1:
        reversed_outs = [activation_func(r) for r in reversed_outs]
        # Apply gradient.
        if n_ins > 1:
            ret = [keras_layers.Add()([tf.gradients(Ys_wo_relu, i, grad_ys=r)[0] for r in reversed_outs]) for i
                   in ins]
        else:
            ret = keras_layers.Add()([tf.gradients(Ys_wo_relu, ins, grad_ys=r)[0] for r in reversed_outs])
    else:
        reversed_outs = activation_func(reversed_outs)
        # Apply gradient.
        if n_ins > 1:
            ret = [tf.gradients(outs, i, grad_ys=reversed_outs)[0] for i in ins]
        else:
            ret = tf.gradients(Ys_wo_relu, ins, grad_ys=reversed_outs)[0]
    return ret

@tf.function
def final_deconvnet_explanation(ins, layer_func, layer_func_wo_relu, activation_func, final_mapping, out_func, reversed_outs, n_ins, n_outs, neuron_selection, r_init):

    outs = tf.squeeze(out_func(ins, layer_func, final_mapping, neuron_selection, r_init))

    # Apply relus conditioned on backpropagated values.
    Ys_wo_relu = tf.squeeze(out_func(ins, layer_func_wo_relu, final_mapping, neuron_selection, r_init))

    if n_outs > 1:
        reversed_outs = [activation_func(r) for r in reversed_outs]
        # Apply gradient.
        if n_ins > 1:
            ret = [keras_layers.Add()([tf.gradients(Ys_wo_relu, i, grad_ys=r)[0] for r in reversed_outs]) for i
                   in ins]
        else:
            ret = keras_layers.Add()([tf.gradients(Ys_wo_relu, ins, grad_ys=r)[0] for r in reversed_outs])
    else:
        reversed_outs = activation_func(reversed_outs)
        # Apply gradient.
        if n_ins > 1:
            ret = [tf.gradients(outs, i, grad_ys=reversed_outs)[0] for i in ins]
        else:
            ret = tf.gradients(Ys_wo_relu, ins, grad_ys=reversed_outs)[0]
    return ret

@tf.function
def guidedbackprop_explanation(ins, layer_func, activation_func, out_func, reversed_outs, n_ins, n_outs):

    #print("TRACING GUIDEDBACKPROP")

    outs = out_func(ins, layer_func)

    if n_outs > 1:
        reversed_outs = [activation_func(r) for r in reversed_outs]
        # Apply gradient.
        if n_ins > 1:
            ret = [keras_layers.Add()([tf.gradients(outs, i, grad_ys=r)[0] for r in reversed_outs]) for i in ins]
        else:
            ret = keras_layers.Add()([tf.gradients(outs, ins, grad_ys=r)[0] for r in reversed_outs])
    else:
        reversed_outs = activation_func(reversed_outs)
        # Apply gradient.
        if n_ins > 1:
            ret = [tf.gradients(outs, i, grad_ys=reversed_outs)[0] for i in ins]
        else:
            ret = tf.gradients(outs, ins, grad_ys=reversed_outs)[0]
    return ret

@tf.function
def final_guidedbackprop_explanation(ins, layer_func, activation_func, final_mapping, out_func, reversed_outs, n_ins, n_outs, neuron_selection, r_init):

    outs = tf.squeeze(out_func(ins, layer_func, final_mapping, neuron_selection, r_init))

    if n_outs > 1:
        reversed_outs = [activation_func(r) for r in reversed_outs]
        # Apply gradient.
        if n_ins > 1:
            ret = [keras_layers.Add()([tf.gradients(outs, i, grad_ys=r)[0] for r in reversed_outs]) for i in ins]
        else:
            ret = keras_layers.Add()([tf.gradients(outs, ins, grad_ys=r)[0] for r in reversed_outs])
    else:
        reversed_outs = activation_func(reversed_outs)
        # Apply gradient.
        if n_ins > 1:
            ret = [tf.gradients(outs, i, grad_ys=reversed_outs)[0] for i in ins]
        else:
            ret = tf.gradients(outs, ins, grad_ys=reversed_outs)[0]
    return ret

#------------LRP
@tf.function
def zrule_explanation(ins, layer_func, out_func, reversed_outs, n_ins, n_outs):

    #print("TRACING Z")

    Zs = out_func(ins, layer_func)
    # print("Tracing Explanation...")
    # Divide incoming relevance by the activations.
    # TODO Leander: consider all cases
    if n_outs > 1:
        tmp = [ilayers.SafeDivide()([tf.reshape(r, Zs.shape), Zs]) for r in reversed_outs]
        # Propagate the relevance to input neurons
        # using the gradient.
        tmp2 = [tf.gradients(Zs, ins, grad_ys=t)[0] for t in tmp]
        ret = keras_layers.Add()([keras_layers.Multiply()([ins, t]) for t in tmp2])
    else:
        tmp = ilayers.SafeDivide()([tf.reshape(reversed_outs, Zs.shape), Zs])
        # Propagate the relevance to input neurons
        # using the gradient.
        tmp2 = tf.gradients(Zs, ins, grad_ys=tmp)[0]
        ret = keras_layers.Multiply()([ins, tmp2])
    return ret

@tf.function
def final_zrule_explanation(ins, layer_func, final_mapping, out_func, reversed_outs, n_ins, n_outs, neuron_selection, r_init):

    Zs = tf.squeeze(out_func(ins, layer_func, final_mapping, neuron_selection, r_init))
    # print("Tracing Explanation...")
    # Divide incoming relevance by the activations.
    # TODO Leander: consider all cases
    if n_outs > 1:
        tmp = [ilayers.SafeDivide()([tf.reshape(r, Zs.shape), Zs]) for r in reversed_outs]
        # Propagate the relevance to input neurons
        # using the gradient.
        tmp2 = [tf.gradients(Zs, ins, grad_ys=t)[0] for t in tmp]
        ret = keras_layers.Add()([keras_layers.Multiply()([ins, t]) for t in tmp2])
    else:
        tmp = ilayers.SafeDivide()([tf.reshape(reversed_outs, Zs.shape), Zs])
        # Propagate the relevance to input neurons
        # using the gradient.
        tmp2 = tf.gradients(Zs, ins, grad_ys=tmp)[0]
        ret = keras_layers.Multiply()([ins, tmp2])
    return ret

#-----

@tf.function
def epsilonrule_explanation(ins, layer_func, out_func, reversed_outs, n_ins, n_outs, epsilon):
    #print("TRACING E")

    Zs = out_func(ins, layer_func)
    # The epsilon rule aligns epsilon with the (extended) sign: 0 is considered to be positive
    prepare_div = keras_layers.Lambda(lambda x: x + (K.cast(K.greater_equal(x, 0), K.floatx()) * 2 - 1) * epsilon)
    # Divide incoming relevance by the activations.
    # TODO Leander: consider all cases
    if n_outs > 1:
        tmp = [ilayers.SafeDivide()([tf.reshape(r, Zs.shape), prepare_div(Zs)]) for r in reversed_outs]
        # Propagate the relevance to input neurons
        # using the gradient.
        tmp2 = [tf.gradients(Zs, ins, grad_ys=t)[0] for t in tmp]
        ret = keras_layers.Add()([keras_layers.Multiply()([ins, t]) for t in tmp2])
    else:
        tmp = ilayers.SafeDivide()([tf.reshape(reversed_outs, Zs.shape), prepare_div(Zs)])
        # Propagate the relevance to input neurons
        # using the gradient.
        tmp2 = tf.gradients(Zs, ins, grad_ys=tmp)[0]
        ret = keras_layers.Multiply()([ins, tmp2])
    return ret

@tf.function
def final_epsilonrule_explanation(ins, layer_func, final_mapping, out_func, reversed_outs, n_ins, n_outs, neuron_selection, r_init, epsilon):

    Zs = tf.squeeze(out_func(ins, layer_func, final_mapping, neuron_selection, r_init))
    # The epsilon rule aligns epsilon with the (extended) sign: 0 is considered to be positive
    prepare_div = keras_layers.Lambda(lambda x: x + (K.cast(K.greater_equal(x, 0), K.floatx()) * 2 - 1) * epsilon)
    # Divide incoming relevance by the activations.
    # TODO Leander: consider all cases
    if n_outs > 1:
        tmp = [ilayers.SafeDivide()([tf.reshape(r, Zs.shape), prepare_div(Zs)]) for r in reversed_outs]
        # Propagate the relevance to input neurons
        # using the gradient.
        tmp2 = [tf.gradients(Zs, ins, grad_ys=t)[0] for t in tmp]
        ret = keras_layers.Add()([keras_layers.Multiply()([ins, t]) for t in tmp2])
    else:
        tmp = ilayers.SafeDivide()([tf.reshape(reversed_outs, Zs.shape), prepare_div(Zs)])
        # Propagate the relevance to input neurons
        # using the gradient.
        tmp2 = tf.gradients(Zs, ins, grad_ys=tmp)[0]
        ret = keras_layers.Multiply()([ins, tmp2])
    return ret

#-----

@tf.function
def wsquarerule_explanation(ins, layer_func, out_func, reversed_outs, n_ins, n_outs):
    #print("TRACING Wsq")

    ones = ilayers.OnesLike()(ins)
    Ys = out_func(ins, layer_func)
    Zs = out_func(ones, layer_func)

    # Weight the incoming relevance.
    # TODO Leander: consider all cases
    if n_outs > 1:
        tmp = [ilayers.SafeDivide()([tf.reshape(r, Zs.shape), Zs]) for r in reversed_outs]
        # Redistribute the relevances along the gradient.
        ret = keras_layers.Add()([tf.gradients(Ys, ins, grad_ys=t)[0] for t in tmp])
    else:
        tmp = ilayers.SafeDivide()([tf.reshape(reversed_outs, Zs.shape), Zs])
        # Redistribute the relevances along the gradient.
        ret = tf.gradients(Ys, ins, grad_ys=tmp)[0]
    return ret

@tf.function
def final_wsquarerule_explanation(ins, layer_func, final_mapping, out_func, reversed_outs, n_ins, n_outs, neuron_selection, r_init):

    ones = ilayers.OnesLike()(ins)
    Ys = tf.squeeze(out_func(ins, layer_func, final_mapping, neuron_selection, r_init))
    Zs = tf.squeeze(out_func(ones, layer_func, final_mapping, neuron_selection, r_init))

    # Weight the incoming relevance.
    # TODO Leander: consider all cases
    if n_outs > 1:
        tmp = [ilayers.SafeDivide()([tf.reshape(r, Zs.shape), Zs]) for r in reversed_outs]
        # Redistribute the relevances along the gradient.
        ret = keras_layers.Add()([tf.gradients(Ys, ins, grad_ys=t)[0] for t in tmp])
    else:
        tmp = ilayers.SafeDivide()([tf.reshape(reversed_outs, Zs.shape), Zs])
        # Redistribute the relevances along the gradient.
        ret = tf.gradients(Ys, ins, grad_ys=tmp)[0]
    return ret

#-----
@tf.function
def alphabetarule_explanation(ins, layer_func_pos, layer_func_neg, out_func, reversed_outs, n_ins, n_outs, alpha, beta):
    #print("TRACING AB")
    keep_positives = keras_layers.Lambda(lambda x: x * K.cast(K.greater(x, 0), K.floatx()))
    keep_negatives = keras_layers.Lambda(lambda x: x * K.cast(K.less(x, 0), K.floatx()))
    ins_pos = keep_positives(ins)
    ins_neg = keep_negatives(ins)

    Zs_pos = out_func(ins_pos, layer_func_pos)
    Zs_neg = out_func(ins_neg, layer_func_neg)
    Zs_pos_n = out_func(ins_pos, layer_func_neg)
    Zs_neg_p = out_func(ins_neg, layer_func_pos)

    # TODO Leander: consider all cases
    times_alpha = keras_layers.Lambda(lambda x: x * alpha)
    times_beta = keras_layers.Lambda(lambda x: x * beta)

    def f(i1, i2, z1, z2, rev):

        Zs = keras_layers.Add()([z1, z2])

        # Divide incoming relevance by the activations.
        if n_outs > 1:
            tmp = [ilayers.SafeDivide()([tf.reshape(r, Zs.shape), Zs]) for r in rev]
            # Propagate the relevance to the input neurons
            # using the gradient
            tmp1 = [tf.gradients(z1, i1, grad_ys=t)[0] for t in tmp]
            tmp2 = [tf.gradients(z2, i2, grad_ys=t)[0] for t in tmp]
            # Re-weight relevance with the input values.
            tmp_1 = [keras_layers.Multiply()([i1, t]) for t in tmp1]
            tmp_2 = [keras_layers.Multiply()([i2, t]) for t in tmp2]
            # combine
            combined = [keras_layers.Add()([a, b]) for a, b in zip(tmp_1, tmp_2)]
        else:
            tmp = ilayers.SafeDivide()([tf.reshape(rev, Zs.shape), Zs])
            # Propagate the relevance to the input neurons
            # using the gradient
            tmp1 = tf.gradients(z1, i1, grad_ys=tmp)[0]
            tmp2 = tf.gradients(z2, i2, grad_ys=tmp)[0]
            # Re-weight relevance with the input values.

            tmp_1 = keras_layers.Multiply()([i1, tmp1])
            tmp_2 = keras_layers.Multiply()([i2, tmp2])
            # combine
            combined = keras_layers.Add()([tmp_1, tmp_2])
        return combined

    # xpos*wpos + xneg*wneg
    activator_relevances = f(ins_pos, ins_neg, Zs_pos, Zs_neg, reversed_outs)

    if beta:  # only compute beta-weighted contributions of beta is not zero
        # xpos*wneg + xneg*wpos
        inhibitor_relevances = f(ins_pos, ins_neg, Zs_pos_n, Zs_neg_p, reversed_outs)
        if n_outs > 1:
            sub = [keras_layers.Subtract()([times_alpha(a), times_beta(b)]) for a, b in
                   zip(activator_relevances, inhibitor_relevances)]
            ret = keras_layers.Add()(sub)
        else:
            ret = keras_layers.Subtract()([times_alpha(activator_relevances), times_beta(inhibitor_relevances)])
        return ret
    else:
        if n_outs > 1:
            ret = keras_layers.Add()(activator_relevances)
        else:
            ret = activator_relevances
        return ret

@tf.function
def final_alphabetarule_explanation(ins, layer_func_pos, layer_func_neg, final_mapping, out_func, reversed_outs, n_ins, n_outs, neuron_selection, r_init, alpha, beta):
    keep_positives = keras_layers.Lambda(lambda x: x * K.cast(K.greater(x, 0), K.floatx()))
    keep_negatives = keras_layers.Lambda(lambda x: x * K.cast(K.less(x, 0), K.floatx()))
    ins_pos = keep_positives(ins)
    ins_neg = keep_negatives(ins)

    Zs_pos = tf.squeeze(out_func(ins_pos, layer_func_pos, final_mapping, neuron_selection, r_init))
    Zs_neg = tf.squeeze(out_func(ins_neg, layer_func_neg, final_mapping, neuron_selection, r_init))
    Zs_pos_n = tf.squeeze(out_func(ins_pos, layer_func_neg, final_mapping, neuron_selection, r_init))
    Zs_neg_p = tf.squeeze(out_func(ins_neg, layer_func_pos, final_mapping, neuron_selection, r_init))

    # TODO Leander: consider all cases
    times_alpha = keras_layers.Lambda(lambda x: x * alpha)
    times_beta = keras_layers.Lambda(lambda x: x * beta)

    def f(i1, i2, z1, z2, rev):

        Zs = keras_layers.Add()([z1, z2])

        # Divide incoming relevance by the activations.
        if n_outs > 1:
            tmp = [ilayers.SafeDivide()([tf.reshape(r, Zs.shape), Zs]) for r in rev]
            # Propagate the relevance to the input neurons
            # using the gradient
            tmp1 = [tf.gradients(z1, i1, grad_ys=t)[0] for t in tmp]
            tmp2 = [tf.gradients(z2, i2, grad_ys=t)[0] for t in tmp]
            # Re-weight relevance with the input values.
            tmp_1 = [keras_layers.Multiply()([i1, t]) for t in tmp1]
            tmp_2 = [keras_layers.Multiply()([i2, t]) for t in tmp2]
            # combine
            combined = [keras_layers.Add()([a, b]) for a, b in zip(tmp_1, tmp_2)]
        else:
            tmp = ilayers.SafeDivide()([tf.reshape(rev, Zs.shape), Zs])
            # Propagate the relevance to the input neurons
            # using the gradient
            tmp1 = tf.gradients(z1, i1, grad_ys=tmp)[0]
            tmp2 = tf.gradients(z2, i2, grad_ys=tmp)[0]
            # Re-weight relevance with the input values.

            tmp_1 = keras_layers.Multiply()([i1, tmp1])
            tmp_2 = keras_layers.Multiply()([i2, tmp2])
            # combine
            combined = keras_layers.Add()([tmp_1, tmp_2])
        return combined

    # xpos*wpos + xneg*wneg
    activator_relevances = f(ins_pos, ins_neg, Zs_pos, Zs_neg, reversed_outs)

    if beta:  # only compute beta-weighted contributions of beta is not zero
        # xpos*wneg + xneg*wpos
        inhibitor_relevances = f(ins_pos, ins_neg, Zs_pos_n, Zs_neg_p, reversed_outs)
        if n_outs > 1:
            sub = [keras_layers.Subtract()([times_alpha(a), times_beta(b)]) for a, b in
                   zip(activator_relevances, inhibitor_relevances)]
            ret = keras_layers.Add()(sub)
        else:
            ret = keras_layers.Subtract()([times_alpha(activator_relevances), times_beta(inhibitor_relevances)])
        return ret
    else:
        if n_outs > 1:
            ret = keras_layers.Add()(activator_relevances)
        else:
            ret = activator_relevances
        return ret
#-----
@tf.function
def alphabetaxrule_explanation(ins, layer_func_pos, layer_func_neg, out_func, reversed_outs, n_ins, n_outs, alpha, beta):
    #print("TRACING ABX")
    keep_positives = keras_layers.Lambda(lambda x: x * K.cast(K.greater(x, 0), K.floatx()))
    keep_negatives = keras_layers.Lambda(lambda x: x * K.cast(K.less(x, 0), K.floatx()))
    ins_pos = keep_positives(ins)
    ins_neg = keep_negatives(ins)

    Zs_pos = out_func(ins_pos, layer_func_pos)
    Zs_neg = out_func(ins_neg, layer_func_neg)
    Zs_pos_n = out_func(ins_pos, layer_func_neg)
    Zs_neg_p = out_func(ins_neg, layer_func_pos)

    # this method is correct, but wasteful
    times_alpha0 = keras_layers.Lambda(lambda x: x * alpha[0])
    times_alpha1 = keras_layers.Lambda(lambda x: x * alpha[1])
    times_beta0 = keras_layers.Lambda(lambda x: x * beta[0])
    times_beta1 = keras_layers.Lambda(lambda x: x * beta[1])

    def f(Xs, Zs, rev):
        # Divide incoming relevance by the activations.
        if n_outs > 1:
            tmp = [ilayers.SafeDivide()([tf.reshape(r, Zs.shape), Zs]) for r in rev]
            # Propagate the relevance to the input neurons
            # using the gradient
            tmp1 = [tf.gradients(Zs, Xs, grad_ys=t)[0] for t in tmp]
            # Re-weight relevance with the input values.
            tmp_1 = [keras_layers.Multiply()([Xs, t]) for t in tmp1]
        else:
            tmp = ilayers.SafeDivide()([tf.reshape(rev, Zs.shape), Zs])
            # Propagate the relevance to the input neurons
            # using the gradient
            tmp1 = tf.gradients(Zs, Xs, grad_ys=tmp)[0]
            # Re-weight relevance with the input values.
            tmp_1 = keras_layers.Multiply()([Xs, tmp1])

        return tmp_1

    # xpos*wpos
    r_pp = f(ins_pos, Zs_pos, reversed_outs)
    # xneg*wneg
    r_nn = f(ins_neg, Zs_neg, reversed_outs)
    # a0 * r_pp + a1 * r_nn
    if n_outs > 1:
        r_pos = [keras_layers.Add()([times_alpha0(pp), times_alpha1(nn)]) for pp, nn in zip(r_pp, r_nn)]
    else:
        r_pos = keras_layers.Add()([times_alpha0(r_pp), times_alpha1(r_nn)])

    # xpos*wneg
    r_pn = f(ins_pos, Zs_pos_n, reversed_outs)
    # xneg*wpos
    r_np = f(ins_neg, Zs_neg_p, reversed_outs)
    # b0 * r_pn + b1 * r_np
    if n_outs > 1:
        r_neg = [keras_layers.Add()([times_beta0(pn), times_beta1(np)]) for pn, np in zip(r_pn, r_np)]
        ret = [keras_layers.Subtract()([a, b]) for a, b in zip(r_pos, r_neg)]
        ret = keras_layers.Add()(ret)
    else:
        r_neg = keras_layers.Add()([times_beta0(r_pn), times_beta1(r_np)])
        ret = keras_layers.Subtract()([r_pos, r_neg])
    return ret

@tf.function
def final_alphabetaxrule_explanation(ins, layer_func_pos, layer_func_neg, final_mapping, out_func, reversed_outs, n_ins, n_outs, neuron_selection, r_init, alpha, beta):
    keep_positives = keras_layers.Lambda(lambda x: x * K.cast(K.greater(x, 0), K.floatx()))
    keep_negatives = keras_layers.Lambda(lambda x: x * K.cast(K.less(x, 0), K.floatx()))
    ins_pos = keep_positives(ins)
    ins_neg = keep_negatives(ins)

    Zs_pos = tf.squeeze(out_func(ins_pos, layer_func_pos, final_mapping, neuron_selection, r_init))
    Zs_neg = tf.squeeze(out_func(ins_neg, layer_func_neg, final_mapping, neuron_selection, r_init))
    Zs_pos_n = tf.squeeze(out_func(ins_pos, layer_func_neg, final_mapping, neuron_selection, r_init))
    Zs_neg_p = tf.squeeze(out_func(ins_neg, layer_func_pos, final_mapping, neuron_selection, r_init))

    # this method is correct, but wasteful
    times_alpha0 = keras_layers.Lambda(lambda x: x * alpha[0])
    times_alpha1 = keras_layers.Lambda(lambda x: x * alpha[1])
    times_beta0 = keras_layers.Lambda(lambda x: x * beta[0])
    times_beta1 = keras_layers.Lambda(lambda x: x * beta[1])

    def f(Xs, Zs, rev):
        # Divide incoming relevance by the activations.
        if n_outs > 1:
            tmp = [ilayers.SafeDivide()([tf.reshape(r, Zs.shape), Zs]) for r in rev]
            # Propagate the relevance to the input neurons
            # using the gradient
            tmp1 = [tf.gradients(Zs, Xs, grad_ys=t)[0] for t in tmp]
            # Re-weight relevance with the input values.
            tmp_1 = [keras_layers.Multiply()([Xs, t]) for t in tmp1]
        else:
            tmp = ilayers.SafeDivide()([tf.reshape(rev, Zs.shape), Zs])
            # Propagate the relevance to the input neurons
            # using the gradient
            tmp1 = tf.gradients(Zs, Xs, grad_ys=tmp)[0]
            # Re-weight relevance with the input values.
            tmp_1 = keras_layers.Multiply()([Xs, tmp1])

        return tmp_1

    # xpos*wpos
    r_pp = f(ins_pos, Zs_pos, reversed_outs)
    # xneg*wneg
    r_nn = f(ins_neg, Zs_neg, reversed_outs)
    # a0 * r_pp + a1 * r_nn
    if n_outs > 1:
        r_pos = [keras_layers.Add()([times_alpha0(pp), times_alpha1(nn)]) for pp, nn in zip(r_pp, r_nn)]
    else:
        r_pos = keras_layers.Add()([times_alpha0(r_pp), times_alpha1(r_nn)])

    # xpos*wneg
    r_pn = f(ins_pos, Zs_pos_n, reversed_outs)
    # xneg*wpos
    r_np = f(ins_neg, Zs_neg_p, reversed_outs)
    # b0 * r_pn + b1 * r_np
    if n_outs > 1:
        r_neg = [keras_layers.Add()([times_beta0(pn), times_beta1(np)]) for pn, np in zip(r_pn, r_np)]
        ret = [keras_layers.Subtract()([a, b]) for a, b in zip(r_pos, r_neg)]
        ret = keras_layers.Add()(ret)
    else:
        r_neg = keras_layers.Add()([times_beta0(r_pn), times_beta1(r_np)])
        ret = keras_layers.Subtract()([r_pos, r_neg])
    return ret

#-----

@tf.function
def boundedrule_explanation(ins, layer_func, layer_func_pos, layer_func_neg, out_func, reversed_outs, n_ins, n_outs, low_param, high_param):
    #print("TRACING bound")
    to_low = keras_layers.Lambda(lambda x: x * 0 + low_param)
    to_high = keras_layers.Lambda(lambda x: x * 0 + high_param)
    low = [to_low(x) for x in ins]
    high = [to_high(x) for x in ins]

    A = out_func(ins, layer_func)
    B = out_func(low, layer_func_pos)
    C = out_func(high, layer_func_neg)

    if n_outs > 1:
        Zs = [keras_layers.Subtract()([a, keras_layers.Add()([b, c])]) for a, b, c in zip(A, B, C)]
        # Divide relevances with the value.
        tmp = [ilayers.SafeDivide()([tf.reshape(r, Zs.shape), Zs]) for r in reversed_outs]
        # Distribute along the gradient.
        tmpA = [tf.gradients(A, ins, grad_ys=t)[0] for t in tmp]
        tmpB = [tf.gradients(B, ins, grad_ys=t)[0] for t in tmp]
        tmpC = [tf.gradients(C, ins, grad_ys=t)[0] for t in tmp]
        ret = keras_layers.Add()(
            [keras_layers.Subtract()([a, keras_layers.Add()([b, c])]) for a, b, c in zip(tmpA, tmpB, tmpC)])
    else:
        Zs = keras_layers.Subtract()([A, keras_layers.Add()([B, C])])
        # Divide relevances with the value.
        tmp = ilayers.SafeDivide()([tf.reshape(reversed_outs, Zs.shape), Zs])
        # Distribute along the gradient.
        tmpA = tf.gradients(A, ins, grad_ys=tmp)[0]
        tmpB = tf.gradients(B, ins, grad_ys=tmp)[0]
        tmpC = tf.gradients(C, ins, grad_ys=tmp)[0]
        ret = keras_layers.Subtract()([tmpA, keras_layers.Add()([tmpB, tmpC])])

    return ret

@tf.function
def final_boundedrule_explanation(ins, layer_func, layer_func_pos, layer_func_neg, final_mapping, out_func, reversed_outs, n_ins, n_outs, neuron_selection, r_init, low_param, high_param):
    to_low = keras_layers.Lambda(lambda x: x * 0 + low_param)
    to_high = keras_layers.Lambda(lambda x: x * 0 + high_param)
    low = [to_low(x) for x in ins]
    high = [to_high(x) for x in ins]

    A = tf.squeeze(out_func(ins, layer_func, final_mapping, neuron_selection, r_init))
    B = tf.squeeze(out_func(low, layer_func_pos, final_mapping, neuron_selection, r_init))
    C = tf.squeeze(out_func(high, layer_func_neg, final_mapping, neuron_selection, r_init))

    if n_outs > 1:
        Zs = [keras_layers.Subtract()([a, keras_layers.Add()([b, c])]) for a, b, c in zip(A, B, C)]
        # Divide relevances with the value.
        tmp = [ilayers.SafeDivide()([tf.reshape(r, Zs.shape), Zs]) for r in reversed_outs]
        # Distribute along the gradient.
        tmpA = [tf.gradients(A, ins, grad_ys=t)[0] for t in tmp]
        tmpB = [tf.gradients(B, ins, grad_ys=t)[0] for t in tmp]
        tmpC = [tf.gradients(C, ins, grad_ys=t)[0] for t in tmp]
        ret = keras_layers.Add()(
            [keras_layers.Subtract()([a, keras_layers.Add()([b, c])]) for a, b, c in zip(tmpA, tmpB, tmpC)])
    else:
        Zs = keras_layers.Subtract()([A, keras_layers.Add()([B, C])])
        # Divide relevances with the value.
        tmp = ilayers.SafeDivide()([tf.reshape(reversed_outs, Zs.shape), Zs])
        # Distribute along the gradient.
        tmpA = tf.gradients(A, ins, grad_ys=tmp)[0]
        tmpB = tf.gradients(B, ins, grad_ys=tmp)[0]
        tmpC = tf.gradients(C, ins, grad_ys=tmp)[0]
        ret = keras_layers.Subtract()([tmpA, keras_layers.Add()([tmpB, tmpC])])

    return ret

#-----

@tf.function
def zplusfastrule_explanation(ins, layer_func, out_func, reversed_outs, n_ins, n_outs):
    #print("TRACING Z+fast")

    Zs = out_func(ins, layer_func)
    if n_outs > 1:
        # Divide incoming relevance by the activations.
        tmp = [ilayers.SafeDivide()([tf.reshape(r, Zs.shape), Zs]) for r in reversed_outs]
        # Propagate the relevance to input neurons
        # using the gradient.
        tmp_1 = [tf.gradients(Zs, ins, grad_ys=t)[0] for t in tmp]
        # Re-weight relevance with the input values.
        ret = keras_layers.Add()([keras_layers.Multiply()([ins, t]) for t in tmp_1])
    else:
        # Divide incoming relevance by the activations.
        tmp = ilayers.SafeDivide()([tf.reshape(reversed_outs, Zs.shape), Zs])
        # Propagate the relevance to input neurons
        # using the gradient.
        tmp_1 = tf.gradients(Zs, ins, grad_ys=tmp)[0]
        # Re-weight relevance with the input values.
        ret = keras_layers.Multiply()([ins, tmp_1])

    return ret

@tf.function
def final_zplusfastrule_explanation(ins, layer_func, final_mapping, out_func, reversed_outs, n_ins, n_outs, neuron_selection, r_init):

    Zs = tf.squeeze(out_func(ins, layer_func, final_mapping, neuron_selection, r_init))
    if n_outs > 1:
        # Divide incoming relevance by the activations.
        tmp = [ilayers.SafeDivide()([tf.reshape(r, Zs.shape), Zs]) for r in reversed_outs]
        # Propagate the relevance to input neurons
        # using the gradient.
        tmp_1 = [tf.gradients(Zs, ins, grad_ys=t)[0] for t in tmp]
        # Re-weight relevance with the input values.
        ret = keras_layers.Add()([keras_layers.Multiply()([ins, t]) for t in tmp_1])
    else:
        # Divide incoming relevance by the activations.
        tmp = ilayers.SafeDivide()([tf.reshape(reversed_outs, Zs.shape), Zs])
        # Propagate the relevance to input neurons
        # using the gradient.
        tmp_1 = tf.gradients(Zs, ins, grad_ys=tmp)[0]
        # Re-weight relevance with the input values.
        ret = keras_layers.Multiply()([ins, tmp_1])

    return ret

#-----
@tf.function
def gammarule_explanation(ins, layer_func, layer_func_pos, out_func, reversed_outs, n_ins, n_outs, gamma):
    #print("TRACING Gamma")
    keep_positives = keras_layers.Lambda(lambda x: x * K.cast(K.greater(x, 0), K.floatx()))
    ins_pos = keep_positives(ins)
    Zs_pos = out_func(ins_pos, layer_func_pos)
    Zs_act = out_func(ins, layer_func)
    Zs_pos_act = out_func(ins_pos, layer_func)
    Zs_act_pos = out_func(ins, layer_func_pos)

    # this method is correct, but wasteful
    times_gamma = keras_layers.Lambda(lambda x: x * gamma)

    def f(i1, i2, z1, z2, rev):

        Zs = keras_layers.Add()([z1, z2])

        # Divide incoming relevance by the activations.
        if n_outs > 1:
            tmp = [ilayers.SafeDivide()([tf.reshape(r, Zs.shape), Zs]) for r in rev]
            # Propagate the relevance to the input neurons
            # using the gradient
            tmp1 = [tf.gradients(z1, i1, grad_ys=t)[0] for t in tmp]
            tmp2 = [tf.gradients(z2, i2, grad_ys=t)[0] for t in tmp]
            # Re-weight relevance with the input values.
            tmp_1 = [keras_layers.Multiply()([i1, t]) for t in tmp1]
            tmp_2 = [keras_layers.Multiply()([i2, t]) for t in tmp2]
            # combine
            combined = [keras_layers.Add()([a, b]) for a, b in zip(tmp_1, tmp_2)]
        else:
            tmp = ilayers.SafeDivide()([tf.reshape(rev, Zs.shape), Zs])
            # Propagate the relevance to the input neurons
            # using the gradient
            tmp1 = tf.gradients(z1, i1, grad_ys=tmp)[0]
            tmp2 = tf.gradients(z2, i2, grad_ys=tmp)[0]
            # Re-weight relevance with the input values.

            tmp_1 = keras_layers.Multiply()([i1, tmp1])
            tmp_2 = keras_layers.Multiply()([i2, tmp2])
            # combine
            combined = keras_layers.Add()([tmp_1, tmp_2])
        return combined

    # xpos*wpos + xact*wact
    activator_relevances = f(ins_pos, ins, Zs_pos, Zs_act, reversed_outs)
    # xpos*wact + xact*wpos
    all_relevances = f(ins_pos, ins, Zs_pos_act, Zs_act_pos, reversed_outs)

    if n_outs > 1:
        sub = [keras_layers.Subtract()([times_gamma(a), b]) for a, b in zip(activator_relevances, all_relevances)]
        ret = keras_layers.Add()(sub)
    else:
        ret = keras_layers.Subtract()([times_gamma(activator_relevances), all_relevances])
    return ret

@tf.function
def final_gammarule_explanation(ins, layer_func, layer_func_pos, final_mapping, out_func, reversed_outs, n_ins, n_outs, neuron_selection, r_init, gamma):
    keep_positives = keras_layers.Lambda(lambda x: x * K.cast(K.greater(x, 0), K.floatx()))
    ins_pos = keep_positives(ins)
    Zs_pos = tf.squeeze(out_func(ins_pos, layer_func_pos, final_mapping, neuron_selection, r_init))
    Zs_act = tf.squeeze(out_func(ins, layer_func, final_mapping, neuron_selection, r_init))
    Zs_pos_act = tf.squeeze(out_func(ins_pos, layer_func, final_mapping, neuron_selection, r_init))
    Zs_act_pos = tf.squeeze(out_func(ins, layer_func_pos, final_mapping, neuron_selection, r_init))

    # this method is correct, but wasteful
    times_gamma = keras_layers.Lambda(lambda x: x * gamma)

    def f(i1, i2, z1, z2, rev):

        Zs = keras_layers.Add()([z1, z2])

        # Divide incoming relevance by the activations.
        if n_outs > 1:
            tmp = [ilayers.SafeDivide()([tf.reshape(r, Zs.shape), Zs]) for r in rev]
            # Propagate the relevance to the input neurons
            # using the gradient
            tmp1 = [tf.gradients(z1, i1, grad_ys=t)[0] for t in tmp]
            tmp2 = [tf.gradients(z2, i2, grad_ys=t)[0] for t in tmp]
            # Re-weight relevance with the input values.
            tmp_1 = [keras_layers.Multiply()([i1, t]) for t in tmp1]
            tmp_2 = [keras_layers.Multiply()([i2, t]) for t in tmp2]
            # combine
            combined = [keras_layers.Add()([a, b]) for a, b in zip(tmp_1, tmp_2)]
        else:
            tmp = ilayers.SafeDivide()([tf.reshape(rev, Zs.shape), Zs])
            # Propagate the relevance to the input neurons
            # using the gradient
            tmp1 = tf.gradients(z1, i1, grad_ys=tmp)[0]
            tmp2 = tf.gradients(z2, i2, grad_ys=tmp)[0]
            # Re-weight relevance with the input values.

            tmp_1 = keras_layers.Multiply()([i1, tmp1])
            tmp_2 = keras_layers.Multiply()([i2, tmp2])
            # combine
            combined = keras_layers.Add()([tmp_1, tmp_2])
        return combined

    # xpos*wpos + xact*wact
    activator_relevances = f(ins_pos, ins, Zs_pos, Zs_act, reversed_outs)
    # xpos*wact + xact*wpos
    all_relevances = f(ins_pos, ins, Zs_pos_act, Zs_act_pos, reversed_outs)

    if n_outs > 1:
        sub = [keras_layers.Subtract()([times_gamma(a), b]) for a, b in zip(activator_relevances, all_relevances)]
        ret = keras_layers.Add()(sub)
    else:
        ret = keras_layers.Subtract()([times_gamma(activator_relevances), all_relevances])
    return ret

