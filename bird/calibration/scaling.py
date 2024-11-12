def scale_x(inp, min_, max_):
    return (inp - min_) / (max_ - min_)


def scale_par(inp, min_, max_):
    return scale_x(inp, min_, max_)


def scale_y(inp, mean_, scale_):
    return (inp - mean_) / (max_ - min_)


def unscale_x(inp, min_, max_):
    return inp * (max_ - min_) + min_


def unscale_par(inp, min_, max_):
    return unscale_x(inp, min_, max_)


def unscale_y(inp, mean_, scale_):
    return inp * scale_ + mean_
