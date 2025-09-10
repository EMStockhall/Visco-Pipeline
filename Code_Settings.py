# This is the classfile for the Code_Settings class
# This class is used to store the settings for the Visco-Pipeline

class pipeline_settings:
    def __init__(self):
        import numpy as np
        import os
        # Get the directory where the current file is located
        file_directory = os.path.dirname(os.path.abspath(__file__))

        # Change the current working directory to the file's directory
        os.chdir(file_directory)

        self.base_directory = os.getcwd()

        #Subfolder containing all the required files
        self.fem_folder = 'fem'
        self.exp_folder = 'exp'
        self.results_folder = 'results'
        
        # Initialize settings

        # File Names
        self.datfile = "v_viscPipe_ring" # DAT file name for MARC Input
        self.expfile = "v_viscPipe_ring.txt" # Numpy file containing the experimental data

        # FEM Parameters
        self.maxFemInc = 21 # The increment at which the max force ocurrs in the FEM simulation
        self.indName = "ringGrip" # The name of the indenter/pulley in the FEM Sim
        self.forceMul = 4 # The force multiplier for the FEM simulation (IE, was it quater or 8th symetry and the force needs to be multiplied by 4 or 8 etc)
        self.flipFemForce = True # If True, the force is flipped in the FEM simulation (IE, if the force is negative, it is flipped to positive)
        self.pronyTerms = 2 # The number of prony terms to use in the FEM simulation
        self.startingpoints = np.array([0.005, 1.954, 0.003488, 700.0])


        # Experimental Parameters
        self.maxExpForce = 100 # The time incremement at which the max force ocurrs in the experimental data

        # Other Settings
        self.normData = True # Normalise the force data for both datasets, if False, the self.normDataFem and self.normDataExp should just be set to None
        self.normDataFem = self.maxFemInc # Can be set to any point, but it is recommended to set it to the max force in the FEM simulation
        self.normDataExp = self.maxExpForce # Can be set to any point, but it is recommended to set it to the max force in the experimental data

        self.saveFinalFem = False # If True, the final FEM simulation is saved to a file

        self.useWholeLoad = False # If True, the whole load history is used to calculate the error function instead of just the hold portion. If false, will start at the max force and go to the end of the sample time.
        self.useWholeLoadStartFem = None # The start point of the whole load history in the FEM simulation
        self.useWholeLoadStartExp = None

        # Sampling Strategy
        self.sampleStrat = "linlog" # The sampling strategy to use, can be "linear" or "log" NBNBNBNBNBNBNBNBNB ADD MORE STRATEGIES IE LINEAR AND LOG!!!!!!! SHOULD HELP WITH BIAS IN THE FRONT AND BACK
        self.samplePoints = 10000 # The number of points along the curve to sample
        self.sampleTime = 1800 # The time to sample the data over, in seconds
        self.error_method = "RMSE" # The error method to use, can be "RMSE" or "MAE"

        # DOT Settings
        self.optimiser_type = "MMFD" # The type of optimiser to use, can be "MMFD", "SLP" or "SQP"
        self.maxdotIterations = 300
        self.nCons = 1 + 2*(self.pronyTerms - 1) # The number of constraints to use in the DOT algorithm
        self.FDCH = 0.005
        self.FDCHM = 0.0005
        dotlowerBound = 0.000001
        self.lowerBound = np.ones((2*self.pronyTerms, 1))*dotlowerBound
        self.lowerBound = self.lowerBound.flatten()
        upperBound1 = 0.999999
        upperBound2 = 10000000
        self.upperBound = np.ones((2*self.pronyTerms, 1))
        for i in range(self.pronyTerms):
            self.upperBound[i*2] = upperBound1
            self.upperBound[i*2 + 1] = upperBound2
        self.upperBound = self.upperBound.flatten()
        self.scalingStrategy = "linln" # The scaling strategy to use, can be "linear" or "log" or "linln"


        
        # Scaling the bounds and starting points to have values in the same order of magnitude
        # This is done to avoid numerical issues with the DOT algorithm

        # Uncomment the next line to use the log10 of the bounds and starting points
        if self.scalingStrategy == "log":
            self.lowerBound = np.log10(self.lowerBound)
            self.upperBound = np.log10(self.upperBound)
            self.startingpoints = np.log10(self.startingpoints)

        # Uncomment the next line to use the linear scaling of the bounds and starting points
        if self.scalingStrategy == "linear":
            self.weights = np.array([0.131, 89.271, 0.0852, 1365.0])
            self.startingpoints = np.divide(self.startingpoints, self.weights)
            self.lowerBound = np.divide(self.lowerBound, self.weights)
            self.upperBound = np.divide(self.upperBound, self.weights)
        
        if self.scalingStrategy == "linln":
            self.weights = np.array([0.131, 89.271, 0.0852, 1365.0])
            for i in range(self.pronyTerms):
                self.lowerBound[i*2] = np.divide(self.lowerBound[i*2], self.weights[i*2])
                self.lowerBound[i*2 + 1] = np.log(self.lowerBound[i*2 + 1])
                self.upperBound[i*2] = np.divide(self.upperBound[i*2], self.weights[i*2])
                self.upperBound[i*2 + 1] = np.log(self.upperBound[i*2 + 1])

                self.startingpoints[i*2] = np.divide(self.startingpoints[i*2], self.weights[i*2])
                self.startingpoints[i*2 + 1] = np.log(self.startingpoints[i*2 + 1])
            

        
        # Folder handling tags
        numerical_model = self.datfile

        self.nm_ftag = numerical_model
        self.nm_t16  = numerical_model + '.t16'
        self.nm_dat  = numerical_model + '.dat'
        self.nm_sts  = numerical_model + '.sts'