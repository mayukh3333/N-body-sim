# N-body-sim
#### Equations
 
 An object at position $\mathbf r_0$ will experience a vector acceleration $\mathbf a$ due to another body with mass $m_1$ at $\mathbf r_1$
 
 $$
 \mathbf a = \frac{G m_1}{|\mathbf r_1- \mathbf r_0|^3}(\mathbf r_1 - \mathbf r_0)
 $$
 where 
 
 If there are multiple bodies, we add the accelerations vectorially:
 $$
 \mathbf a  = \sum_{i=1}^N \frac{G m_i}{|\mathbf r_i- \mathbf r_0|^3}(\mathbf r_i- \mathbf r_0)
 $$
 
 Done using various ways like Eulers method RK2 and RK4. Option to call any method and automatic graph ploting
