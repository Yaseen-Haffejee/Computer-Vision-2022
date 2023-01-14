from classes import *

puzzle = Puzzle(MATCH_IMGS)
corner_piece = puzzle.pieces[3]

# Create our canvas with the necessary size
canvas = np.zeros((800,700,3))

visited = [] #keep track of visited nodes.
queue = [corner_piece]  #initialize a queue with root node

corner_piece.insert(canvas)
corner_piece.inserted = True  #insert root node
plt.imshow(canvas)
plt.savefig("canvasWith1pieces.jpg")
plt.show()

#RUN BFS
def bfs(visited, node, queue):
	visited.append(node)
	countPuzzle = 1 #initialise puzzle inserted counter
	while queue:
		piece = queue.pop(0) #pop puzzle
		count = 0 #check edges
		for neighbour in piece.return_edge():  #for neighbours
			connectedEdge = neighbour.connected_edge
			if connectedEdge ==None: #if it has no connected edge then skip
				print("skip")
			else:
				connectedEdge = neighbour.connected_edge.parent_piece #find parent piece
				if connectedEdge not in visited: #if it has not been visited
					visited.append(connectedEdge) #add to lists
					queue.append(connectedEdge)

					connectedEdge.insert(canvas) #add to canvas
					connectedEdge.inserted = True
					countPuzzle = countPuzzle + 1 #increase puzzle counter
					# plt.imshow(canvas)
					# plt.show()
					# plt.close()
					
					if((countPuzzle > 0 and countPuzzle < 6) or countPuzzle > 43): #save necessary images for output of lab with necessary puzzles
						plt.imshow(canvas)
						plt.savefig("canvasWith"+str(countPuzzle)+"pieces.jpg")
						plt.show()

			count = count + 1 #count edges
			if count == 4:
				break

#call BFS
bfs(visited, corner_piece, queue)