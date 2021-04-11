import os

a = 1

liste = os.listdir("./Photos/Pub")

for i in range(len(liste)) :
	#fich1 = "./Photos/Logo/tf1" + str(a) + ".png"
	fich2 = "./PhotosCharcoal/Pub/pub" + str(a) + "coal.png"
	commande = "magick convert " + "./Photos/Pub/" + str(liste[i]) + " -charcoal 1 " + fich2
	try:
		os.system('cmd /c' + '"' + commande + '"')
	except:
		print("no")

	a = a + 1