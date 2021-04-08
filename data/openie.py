from pyopenie import OpenIE5
extractor = OpenIE5('http://localhost:9000')
extractions = extractor.extract("Barack Obama gave his speech to thousands of people.")