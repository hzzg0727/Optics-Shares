from RayTracing import *

# step n (lower to higher)
sys = System()
sys.setRegion(-5, 5, -5, 5, 0, 10)
sys.setStep(0.1, 0.1, 0.1)
sys.addCubic(-5, 5, -5, 5, 0, 2, 1)
sys.addCubic(-5, 5, -5, 5, 2, 4, 1.5)
sys.addCubic(-5, 5, -5, 5, 4, 6, 2)
sys.addCubic(-5, 5, -5, 5, 6, 8, 2.5)
sys.addCubic(-5, 5, -5, 5, 8, 10, 3)
sys.addLight(4, 0, -0.9, 0, 0)
sys.addLight(4, 0, -0.7, 0, 0)
sys.addLight(4, 0, -0.5, 0, 0)
sys.trace()

sys.drawContinuous(elev=0, azim=-90)

sys.drawDiscrete(elev=10, azim=45, grid_number=32)
