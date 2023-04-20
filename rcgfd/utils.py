from matplotlib.lines import Line2D

def global_legend(fig, labels, linewidth=10, color_start=0, **kwargs):

    lines = [Line2D([0], [0], color=f"C{i+color_start}", linewidth=linewidth) for i, _ in enumerate(labels)]

    if labels[-1].lower() == "persistence":
        lines.pop()
        lines.append(Line2D([0], [0], color=f"gray", linewidth=linewidth))

    loc = kwargs.pop("loc", "center")
    bbox_to_anchor = kwargs.pop("bbox_to_anchor", (0.5, -0.075))
    ncol = kwargs.pop("ncol", len(labels))
    frameon = kwargs.pop("frameon", False)
    fig.legend(
            lines,
            labels,
            loc=loc,
            bbox_to_anchor=bbox_to_anchor,
            ncol=ncol,
            frameon=frameon,
            **kwargs
            )
    return
