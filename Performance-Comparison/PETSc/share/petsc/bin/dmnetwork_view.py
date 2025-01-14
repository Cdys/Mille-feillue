#!/usr/bin/env python3

global nextWindowOffset
nextWindowOffset = 50

# Parses a color string into an RGBA tuple to use with matplotlib
def parseColor(color, defaultValue = (0, 0, 0, 1)):
	# We only accept string values for parsing
	if not isinstance(color, str):
		return defaultValue
	else:
		# Currently only HTML format colors are accepted, ie. #RRGGBB
		if color[0] == '#':
			rgb = tuple(int(color[i:i+2], 16) for i in (1, 3, 5))
			return (rgb[0] / 255, rgb[1] / 255, rgb[2] / 255, 1)
		else:
			return defaultValue

# Parses an ID value to a string *consistently*
def parseID(idval):
	# If float has no fractional part format it as an integer instead of leaving trailing zeros
	if isinstance(idval, float):
		if idval % 1 == 0:
			return str(int(idval))
	return str(idval)


class DisplayOptions:
	def __init__(self,args):
	# Parse any set node or edge colors
		self.nodeColor = None
		self.edgeColor = None
		self.nodeTitleColor = None
		self.edgeTitleColor = None

		if 'set_node_color' in args:
			self.nodeColor = parseColor(args.set_node_color, None)
		if 'set_edge_color' in args:
			self.edgeColor = parseColor(args.set_edge_color, None)

		if 'set_node_title_color' in args:
			self.nodeTitleColor = parseColor(args.set_node_title_color, (1, 1, 1, 1))
		if 'set_edge_title_color' in args:
			self.edgeTitleColor = parseColor(args.set_edge_title_color)

		self.noNodes = 'no_nodes' in args and args.no_nodes
		self.setTitle = args.set_title if 'set_title' in args else None
		self.noNodeLabels = 'no_node_labels' in args and args.no_node_labels
		self.noEdgeLabels = 'no_edge_labels' in args and args.no_edge_labels


# Class for holding the properties of a node
class Node:
	def __init__(self, row, opts: DisplayOptions):
		# Set our ID and rank
		self.id = parseID(row['ID'])
		self.rank = int(row['Rank'])
		
		# Set our position
		x = float(row['X'])
		if np.isnan(x):
			x = 0
		y = float(row['Y'])
		if np.isnan(y):
			y = 0
		z = float(row['Z'])
		if np.isnan(z):
			z = 0
		self.position = (x,y,z)

		# Set name and color, defaulting to a None name if not specified
		self.name = row['Name']
		if not isinstance(self.name, str):
			if np.isnan(self.name):
				self.name = None
			else:
				self.name = str(self.name)

		self.color = opts.nodeColor or parseColor(row['Color'])

# Class for holding the properties of an edge
class Edge:
	def __init__(self, row, opts: DisplayOptions, nodes):
		# Set our ID and rank
		self.id = parseID(row['ID'])
		self.rank = int(row['Rank'])

		# Determine our starting and ending nodes from the X and Y properties
		start = parseID(row['X'])
		if not start in nodes:
			raise KeyError("No such node \'" + str(start) + "\' for start of edge \'" + str(self.id) + '\'')
		self.startNode = nodes[start]
		end = parseID(row['Y'])
		if not end in nodes:
			raise KeyError ("No such node \'" + str(end) + "\' for end of edge \'" + str(self.id) + '\'')
		self.endNode = nodes[end]

		# Set name and color, defaulting to a None name if not specified
		self.name = row['Name']
		if not isinstance(self.name, str):
			if np.isnan(self.name):
				self.name = None
			else:
				self.name = str(self.name)

		self.color = opts.edgeColor or parseColor(row['Color'], (0.5, 0.5, 0.5, 1))


# Class for holding the data for a rank
class Rank:
	def __init__(self, index):
		self.id = index
		self.nodes = {}
		self.edges = {}

	def display(self, opts: DisplayOptions, title):
		# Create Numpy arrays for node and edge positions and colors
		nodePositions = np.zeros((len(self.nodes), 2))
		nodeColors = np.zeros((len(self.nodes), 4))
		edgeSegments = np.zeros((len(self.edges), 2, 2))
		edgeColors = np.zeros((len(self.edges), 4))

		# Copy node positions and colors to the arrays
		i = 0
		for node in self.nodes.values():
			nodePositions[i] = node.position[0], node.position[1]
			nodeColors[i] = node.color
			i += 1

		# Copy edge positions and colors to the arrays
		i = 0
		for edge in self.edges.values():
			start = edge.startNode.position
			end = edge.endNode.position
			edgeSegments[i] = [
				(start[0], start[1]),
				(end[0], end[1])
			]
			edgeColors[i] = edge.color
			i += 1

		# Start the figure for this rank
		fig = plt.figure("Rank " + str(self.id) if self.id >= 0 else "Global")
		try:
			global nextWindowOffset
			offset = nextWindowOffset
			nextWindowOffset += 50

			window = fig.canvas.manager.window
			backend = matplotlib.get_backend()
			if backend == 'TkAgg':
				window.wm_geometry("+%d+%d" % (offset, offset))
			elif backend == 'WXAgg':
				window.SetPosition(offset, offset)
			else:
				window.move(offset, offset)
		except Exception:
			pass
		# Get axis for the plot
		axis = fig.add_subplot()

		# Set the title of the plot if specified
		if opts.setTitle:
			title = (opts.setTitle, (0, 0, 0, 1))
		if title is None:
			title = ("Network", (0, 0, 0, 1))
		if self.id != -1:
			title = (title[0] + " (Rank " + str(self.id) + ")", title[1])
		axis.set_title(title[0], color=title[1])

		# Add a line collection to the axis for the edges
		axis.add_collection(LineCollection(
			segments=edgeSegments,
			colors=edgeColors,
			linewidths=2
		))

		if not opts.noNodes:
			# Add a circle collection to the axis for the nodes
			axis.add_collection(CircleCollection(
				sizes=np.ones(len(self.nodes)) * (20 ** 2),
				offsets=nodePositions,
				transOffset=axis.transData,
				facecolors=nodeColors,
				# Place above the lines
				zorder=3
			))

		if not opts.noNodeLabels and not opts.noNodes:
			# For each node, plot its name at the center of its point
			for node in self.nodes.values():
				if node.name is not None:
					axis.text(
						x=node.position[0], y=node.position[1],
						s=node.name,
						# Center text vertically and horizontally
						va='center', ha='center',
						# Make sure the text is clipped within the plot area
						clip_on=True,
						color=opts.nodeTitleColor
					)

		if not opts.noEdgeLabels:
			# For each edge, plot its name at the center of the line segment
			for edge in self.edges.values():
				if edge.name is not None:
					axis.text(
						x=(edge.startNode.position[0]+edge.endNode.position[0])/2,
						y=(edge.startNode.position[1]+edge.endNode.position[1])/2,
						s=edge.name,
						va='center', ha='center',
						clip_on=True,
						color=opts.edgeTitleColor
					)

		# Scale the plot to the content
		axis.autoscale()

def main(args):
	# Parse display options from arguments
	opts = DisplayOptions(args)

	# The set of ranks
	ranks = { -1: Rank(-1) }

	maxRank = None

	def getRank(rank: int):
		if rank in ranks:
			return ranks[rank]
		else:
			r = Rank(rank)
			ranks[rank] = r
			return r
		
	globalRank = ranks[-1]

	# Global variable storing a title to use or None
	title = None

	# Read each file passed in arguments
	for filename in args.filenames:
		try:
			# Read the data from the supplied CSV file
			data = pd.read_csv(filename, skipinitialspace=True)
			# Iterate each row of data in the file
			for i,row in data.iterrows():
				# Switch based on the type of the entry
				type = row['Type']
				if type == 'Type':
					# If we encounter 'Type' again it is a duplicate header and should be skipped
					continue
				elif type == 'Title':
					# Set the title based on name and color
					titleColor = parseColor(row['Color'])
					title = (row['Name'], titleColor)
				elif type == 'Node':
					# Register a new node
					node = Node(row, opts)
					globalRank.nodes[node.id] = node
					r = getRank(node.rank)
					if r is not None:
						r.nodes[node.id] = node
				elif type == 'Edge':
					# Register a new edge
					edge = Edge(row, opts, globalRank.nodes)
					globalRank.edges[edge.id] = edge
					r = getRank(node.rank)
					if r is not None:
						edge = Edge(row, opts, r.nodes)
						r.edges[edge.id] = edge
		except Exception as e:
			print("Warning! Could not read file \"" + filename + "\": " + str(e))
			traceback.print_exc(file=sys.stdout)
			exit(-1)

	# Show the plot
	if not args.no_display:
		# Generate figures using ranks
		if not args.no_combined_plot:
			globalRank.display(opts, title)
		if args.draw_rank_range:
			ranges = str(args.draw_rank_range).split(',')
			for rangeStr in ranges:
				if '-' in rangeStr != -1:
					limits = rangeStr.split('-')
					for rank in range(int(limits[0]), int(limits[1])+1):
						if rank in ranks:
							ranks[rank].display(opts, title)
				else:
					rank = int(rangeStr)
					if rank in ranks:
						ranks[rank].display(opts, title)
		elif args.draw_all_ranks:
			for rank in ranks:
				if rank != -1:
					ranks[rank].display(opts, title)
			

		# Delay based on options
		if args.display_time is not None:
			plt.show(block=False)
			plt.pause(float(args.display_time))
			# Try to bring the window to front since we are displaying for a limited time
			try:
				window = plt.get_current_fig_manager().window
				window.activateWindow()
				window.raise_()
			except AttributeError:
				pass
		else:
			plt.show()
		plt.close()


if __name__ == "__main__":
	try:
		from argparse import ArgumentParser
		# Construct the argument parse and parse the program arguments
		argparser = ArgumentParser(
			prog='dmnetwork_view.py',
			description="Displays a CSV file generated from a DMNetwork using matplotlib"
		)
		argparser.add_argument('filenames', nargs='+')
		argparser.add_argument('-t', '--set-title', metavar='TITLE', action='store', help="Sets the title for the generated plot, overriding any title set in the source file")
		argparser.add_argument('-nnl', '--no-node-labels', action='store_true', help="Disables labeling nodes in the generated plot")
		argparser.add_argument('-nel', '--no-edge-labels', action='store_true', help="Disables labeling edges in the generated plot")
		argparser.add_argument('-nc', '--set-node-color', metavar='COLOR', action='store', help="Sets the color for drawn nodes, overriding any per-node colors")
		argparser.add_argument('-ec', '--set-edge-color', metavar='COLOR', action='store', help="Sets the color for drawn edges, overriding any per-edge colors")
		argparser.add_argument('-ntc', '--set-node-title-color', metavar='COLOR', action='store', help="Sets the color for drawn node titles, overriding any per-node colors")
		argparser.add_argument('-etc', '--set-edge-title-color', metavar='COLOR', action='store', help="Sets the color for drawn edge titles, overriding any per-edge colors")
		argparser.add_argument('-nd', '--no-display', action='store_true', help="Disables displaying the figure, but will parse as normal")
		argparser.add_argument('-tx', '--test-execute', action='store_true', help="Returns from the program immediately, used only to test run the script")
		argparser.add_argument('-dt', '--display-time', metavar='SECONDS', action='store', help="Sets the time to display the figure in seconds before automatically closing the window")
		argparser.add_argument('-dar', '--draw-all-ranks', action='store_true', help="Draws each rank's network in a separate figure")
		argparser.add_argument('-ncp', '--no-combined-plot', action='store_true', help="Disables drawing the combined network figure")
		argparser.add_argument('-drr', '--draw-rank-range', action='store', help="Specifies a comma-separated list of rank numbers or ranges to display, eg. \'1,3,5-9\'")
		argparser.add_argument('-nn', '--no-nodes', action='store_true', help="Disables displaying the nodes")
		args = argparser.parse_args()

		if not args.test_execute:
			import pandas as pd
			import numpy as np
			import matplotlib
			import matplotlib.pyplot as plt
			from matplotlib.collections import CircleCollection, LineCollection
			import traceback
			import sys

			main(args)
	except ImportError as error:
		print("Missing import: " + str(error))
		exit(-1)





