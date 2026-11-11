import { render, screen, act } from '@testing-library/react';
import userEvent from '@testing-library/user-event';
import { beforeEach, describe, expect, it, vi } from 'vitest';
import { PWAInstallButton } from './PWAInstallButton';

describe('PWAInstallButton', () => {
  beforeEach(() => {
    vi.restoreAllMocks();
    Object.defineProperty(window, 'matchMedia', {
      writable: true,
      value: vi.fn().mockImplementation(query => ({
        matches: false,
        media: query,
        onchange: null,
        addListener: vi.fn(),
        removeListener: vi.fn(),
        addEventListener: vi.fn(),
        removeEventListener: vi.fn(),
        dispatchEvent: vi.fn(),
      })),
    });
  });

  it('renders nothing by default when not installable and not iOS', () => {
    const { container } = render(<PWAInstallButton />);
    expect(container).toBeEmptyDOMElement();
  });

  it('renders install button when beforeinstallprompt event is dispatched', async () => {
    const user = userEvent.setup();
    render(<PWAInstallButton />);

    const promptMock = vi.fn().mockResolvedValue(undefined);
    const beforeInstallEvent = new Event('beforeinstallprompt') as Event & {
      prompt: () => Promise<void>;
      userChoice: Promise<{ outcome: string; platform: string }>;
    };
    beforeInstallEvent.prompt = promptMock;
    beforeInstallEvent.userChoice = Promise.resolve({ outcome: 'accepted', platform: 'web' });

    act(() => {
      window.dispatchEvent(beforeInstallEvent);
    });

    const button = await screen.findByRole('button', { name: 'アプリをインストール' });
    expect(button).toBeInTheDocument();

    await user.click(button);
    expect(promptMock).toHaveBeenCalledTimes(1);
  });

  it('hides install button when appinstalled event is triggered', async () => {
    render(<PWAInstallButton />);

    const beforeInstallEvent = new Event('beforeinstallprompt') as Event & {
      prompt: () => Promise<void>;
      userChoice: Promise<{ outcome: string; platform: string }>;
    };
    beforeInstallEvent.prompt = vi.fn().mockResolvedValue(undefined);
    beforeInstallEvent.userChoice = new Promise(() => {});

    act(() => {
      window.dispatchEvent(beforeInstallEvent);
    });

    expect(await screen.findByRole('button', { name: 'アプリをインストール' })).toBeInTheDocument();

    act(() => {
      window.dispatchEvent(new Event('appinstalled'));
    });

    expect(screen.queryByRole('button', { name: 'アプリをインストール' })).not.toBeInTheDocument();
  });

  it('renders iOS guide modal when on iOS and not standalone', async () => {
    const originalUserAgent = window.navigator.userAgent;
    Object.defineProperty(window.navigator, 'userAgent', {
      value: 'Mozilla/5.0 (iPhone; CPU iPhone OS 17_0 like Mac OS X)',
      configurable: true,
    });

    const user = userEvent.setup();
    render(<PWAInstallButton />);

    const button = screen.getByRole('button', { name: 'iOSでホーム画面に追加' });
    expect(button).toBeInTheDocument();

    await user.click(button);
    expect(screen.getByText('iPhone / iPad でのインストール')).toBeInTheDocument();
    expect(screen.getByText(/Safari のツールバーにある/)).toBeInTheDocument();

    await user.click(screen.getByRole('button', { name: '案内を閉じる' }));
    expect(screen.queryByText('iPhone / iPad でのインストール')).not.toBeInTheDocument();

    Object.defineProperty(window.navigator, 'userAgent', {
      value: originalUserAgent,
      configurable: true,
    });
  });
});
