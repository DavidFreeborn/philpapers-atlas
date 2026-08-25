import { createRoot } from 'react-dom/client';
import Atlas from '../app/Atlas';
import '../app/globals.css';

const root = document.getElementById('root');
if (!root) throw new Error('Application root not found');

createRoot(root).render(<Atlas />);
