import matplotlib.pyplot as plt
import matplotlib.animation as manim

class FFmpegMP4:
    """
    with rp.viz.save.FFmpegMP4(fig, fps, "video.mp4") as writer:
        ...
        plt.plot()
        ...
        writer.grab_frame()

    writer = rp.viz.save.FFmpegMP4(fig, fps, "video.mp4")
    writer.open()
    ...
    writer.close()
    """
    def __init__(self, fig, fps, path, metadata=None, dpi=100):
        self.fig = fig
        self.path = path
        self.dpi = dpi
        if metadata is None:
            metadata = dict(title="Video", artist="Author")

        ImplWriter = manim.writers["ffmpeg"]
        self.writer = ImplWriter(fps=fps, metadata=metadata)
        self.writer.setup(fig, path)

    def open(self):
        self.wh = self.writer.saving(self.fig, self.path, self.dpi)

    def close(self):
        self.writer.finish()

    def __call__(self):
        return self.__enter__()

    def __enter__(self):
        self.open()
        return self

    def __exit__(self, *args, **kwargs):
        self.close()

    def grab_frame(self):
        return self.writer.grab_frame()
