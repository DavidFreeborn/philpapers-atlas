import type { Metadata } from 'next';
import { Geist, Geist_Mono } from 'next/font/google';
import './globals.css';

const geistSans = Geist({
  variable: '--font-geist-sans',
  subsets: ['latin'],
});

const geistMono = Geist_Mono({
  variable: '--font-geist-mono',
  subsets: ['latin'],
});

export const metadata: Metadata = {
  title: 'PhilPapers Atlas',
  description:
    'Explore 69,400 philosophy papers through an interactive UMAP of SPECTER embeddings and HDBSCAN clusters.',
  openGraph: {
    title: 'PhilPapers Atlas',
    description: 'Explore 69,400 philosophy papers in an interactive semantic map.',
    type: 'website',
    images: [{ url: '/og.png', width: 1734, height: 900, alt: 'PhilPapers Atlas — Explore 69,400 philosophy papers' }],
  },
  twitter: {
    card: 'summary_large_image',
    title: 'PhilPapers Atlas',
    description: 'Explore 69,400 philosophy papers in an interactive semantic map.',
    images: ['/og.png'],
  },
};

export default function RootLayout({
  children,
}: Readonly<{
  children: React.ReactNode;
}>) {
  return (
    <html lang="en">
      <body
        className={`${geistSans.variable} ${geistMono.variable} antialiased`}
      >
        {children}
      </body>
    </html>
  );
}
