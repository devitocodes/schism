import pytest
from schism.boundaryconditions import Basis


class TestBasis:
    """Tests for the Basis object"""

    # Setup and correct answers
    basis_test_config = [(2, 1, 'Basis((-a[0] + x[0])**2*d[2]/2 '
                          + '+ (-a[0] + x[0])*d[1] + d[0])'),
                         (4, 1, 'Basis((-a[0] + x[0])**4*d[4]/24 '
                          + '+ (-a[0] + x[0])**3*d[3]/6 + (-a[0] '
                          + '+ x[0])**2*d[2]/2 + (-a[0] + x[0])*d[1] '
                          + '+ d[0])'),
                         (6, 1, 'Basis((-a[0] + x[0])**6*d[6]/720 '
                          + '+ (-a[0] + x[0])**5*d[5]/120 + (-a[0] '
                          + '+ x[0])**4*d[4]/24 + (-a[0] + x[0])**3*d[3]/6 '
                          + '+ (-a[0] + x[0])**2*d[2]/2 + (-a[0] '
                          + '+ x[0])*d[1] + d[0])'),
                         (2, 2, 'Basis((-a[0] + x[0])**2*d[2, 0]/2 '
                          + '+ (-a[0] + x[0])*(-a[1] + x[1])*d[1, 1] '
                          + '+ (-a[0] + x[0])*d[1, 0] + (-a[1] '
                          + '+ x[1])**2*d[0, 2]/2 + (-a[1] + x[1])*d[0, 1] '
                          + '+ d[0, 0])'),
                         (4, 2, 'Basis((-a[0] + x[0])**4*d[4, 0]/24 '
                          + '+ (-a[0] + x[0])**3*(-a[1] + x[1])*d[3, 1]/6 '
                          + '+ (-a[0] + x[0])**3*d[3, 0]/6 + (-a[0] '
                          + '+ x[0])**2*(-a[1] + x[1])**2*d[2, 2]/4 '
                          + '+ (-a[0] + x[0])**2*(-a[1] + x[1])*d[2, 1]/2 '
                          + '+ (-a[0] + x[0])**2*d[2, 0]/2 + (-a[0] '
                          + '+ x[0])*(-a[1] + x[1])**3*d[1, 3]/6 + (-a[0] '
                          + '+ x[0])*(-a[1] + x[1])**2*d[1, 2]/2 + (-a[0] '
                          + '+ x[0])*(-a[1] + x[1])*d[1, 1] + (-a[0] '
                          + '+ x[0])*d[1, 0] + (-a[1] + x[1])**4*d[0, 4]/24 '
                          + '+ (-a[1] + x[1])**3*d[0, 3]/6 + (-a[1] '
                          + '+ x[1])**2*d[0, 2]/2 + (-a[1] + x[1])*d[0, 1] '
                          + '+ d[0, 0])'),
                         (2, 3, 'Basis((-a[0] + x[0])**2*d[2, 0, 0]/2 '
                          + '+ (-a[0] + x[0])*(-a[1] + x[1])*d[1, 1, 0] '
                          + '+ (-a[0] + x[0])*(-a[2] + x[2])*d[1, 0, 1] '
                          + '+ (-a[0] + x[0])*d[1, 0, 0] + (-a[1] '
                          + '+ x[1])**2*d[0, 2, 0]/2 + (-a[1] + x[1])*(-a[2] '
                          + '+ x[2])*d[0, 1, 1] + (-a[1] + x[1])*d[0, 1, 0] '
                          + '+ (-a[2] + x[2])**2*d[0, 0, 2]/2 + (-a[2] '
                          + '+ x[2])*d[0, 0, 1] + d[0, 0, 0])')]

    @pytest.mark.parametrize('config', basis_test_config)
    def test_basis(self, config):
        """Check correct basis expression is returned"""
        order, ndims, answer = config
        basis = Basis(order, ndims)
        assert str(basis) == answer

    # Setup and correct answers
    deriv_test_config = [(2, 1, (1,), '(-a[0] + x[0])*d[2] + d[1]'),
                         (4, 1, (1,), '(-a[0] + x[0])**3*d[4]/6 + (-a[0] '
                          + '+ x[0])**2*d[3]/2 + (-a[0] + x[0])*d[2] + d[1]'),
                         (6, 1, (4,), '(a[0] - x[0])**2*d[6]/2 - (a[0] '
                          + '- x[0])*d[5] + d[4]'),
                         (2, 2, (0, 1), '(-a[0] + x[0])*d[1, 1] + (-a[1] '
                          + '+ x[1])*d[0, 2] + d[0, 1]'),
                         (2, 2, (1, 0), '(-a[0] + x[0])*d[2, 0] + (-a[1] '
                          + '+ x[1])*d[1, 1] + d[1, 0]'),
                         (4, 2, (2, 1), '-(a[0] - x[0])*d[3, 1] - (a[1] '
                          + '- x[1])*d[2, 2] + d[2, 1]'),
                         (2, 3, (0, 1, 0), '(-a[0] + x[0])*d[1, 1, 0] '
                          + '+ (-a[1] + x[1])*d[0, 2, 0] + (-a[2] '
                          + '+ x[2])*d[0, 1, 1] + d[0, 1, 0]'),
                         (4, 3, (2, 1, 0), '-(a[0] - x[0])*d[3, 1, 0] '
                          + '- (a[1] - x[1])*d[2, 2, 0] - (a[2] '
                          + '- x[2])*d[2, 1, 1] + d[2, 1, 0]')]

    @pytest.mark.parametrize('config', deriv_test_config)
    def test_derivatives(self, config):
        """Check that derivatives are calculated correctly"""
        order, ndims, derivs, answer = config
        basis = Basis(order, ndims)
        deriv_expr = basis.derivative(derivs)
        assert str(deriv_expr) == answer


class TestBC:
    """Tests for the BoundaryCondition object"""


def main():
    """Main function for troubleshooting"""
    print("Hello world!")


if __name__ == "__main__":
    main()
