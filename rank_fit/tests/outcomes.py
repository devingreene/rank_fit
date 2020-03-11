import rank_fit.abilities as abilities
import unittest
import numpy as np
import rank_fit.rankers as rankers
import rank_fit.tournaments as Tours

class TestOutcomes(unittest.TestCase):
    def runTest(self):
        candidates = np.random.randint(0,5,size=(100,5,5))
        for c in candidates:
            with self.subTest(c=c):
                tour = Tours.UserDefinedTournament(c)
                with np.errstate(all='raise'):
                    rankers.PBS_ranker(tour,alpha=0.01)
                    rankers.glicko_ranker(tour)

class TestOutcomeswElo(unittest.TestCase):
    def runTest(self):
        skills = np.random.uniform(-2,2,size=(100,5))
        for s in skills:
            ptab = abilities.standard(s)
            tour = Tours.RandomTournament(ptab,10,keep_history=True)
            with self.subTest(s=s,tour=tour):
                with np.errstate(all='raise'):
                    rankers.PBS_ranker(tour,alpha=0.01)
                    while True:
                        try:
                            rankers.glicko_ranker(tour)
                            break
                        except:
                            pass
                    rankers.elo_ranker(tour,k=32)

if __name__ == '__main__':
    unittest.main()
