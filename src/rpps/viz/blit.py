class BlitManager:
    def __init__(self, fig, animated_artists=()):
        self.fig = fig
        self.canvas = fig.canvas
        self._artists = []
        self._bg = None
        for a in animated_artists:
            self.add_artist(a)
        self.cid = fig.canvas.mpl_connect("draw_event", self.on_draw)
        self.size = fig.get_size_inches()

    def on_draw(self, event): # Handle resizes
        if event is not None:
            if event.canvas != self.canvas:
                raise RuntimeError

        size = self.fig.get_size_inches()
        if (self.size != size).any():
            self.size = size
            print(f"Saving BG")
            self._bg = self.canvas.copy_from_bbox(self.canvas.figure.bbox)
        self._draw_animated()

    def _draw_animated(self):
        for a in self._artists:
            self.fig.draw_artist(a)

    def add_artist(self, art):
        if art.figure != self.canvas.figure:
            raise RuntimeError
        art.set_animated(True)
        self._artists.append(art)

    def update(self):
        if self._bg is None:
            self.on_draw(None)
        self.canvas.restore_region(self._bg)
        self._draw_animated()
        self.canvas.blit(self.canvas.figure.bbox)
        self.canvas.flush_events()
